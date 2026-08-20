#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

class PackedUInt64Reservoir
{
public:
    PackedUInt64Reservoir(uint64_t capacity, uint64_t rng_base)
        : capacity_(capacity), rng_base_(rng_base)
    {
        if (capacity == 0)
            throw std::invalid_argument("packed reservoir capacity must be positive");
        slots_.reserve(capacity);
    }

    py::array_t<uint64_t> offer(
        py::array_t<uint64_t, py::array::c_style> input)
    {
        if (closed_)
            throw std::runtime_error("cannot offer items after the reservoir was drained");
        if (input.ndim() != 1)
            throw std::invalid_argument("packed reservoir input must be one-dimensional");

        const auto input_view = input.unchecked<1>();
        const uint64_t input_count = static_cast<uint64_t>(input_view.shape(0));
        const uint64_t free_slots = capacity_ - slots_.size();
        const uint64_t output_count = input_count > free_slots
                                          ? input_count - free_slots
                                          : 0;
        py::array_t<uint64_t> output(output_count);
        auto output_view = output.mutable_unchecked<1>();

        // Validate the complete deterministic RNG suffix before mutating slots.
        // A restored counter at its limit therefore fails atomically without
        // paying for a reservoir copy or transaction log on the normal path.
        uint64_t preflight_counter = rng_counter_;
        for (uint64_t position = 0; position < output_count; ++position)
            output_view(position) = random_slot(preflight_counter, capacity_);
        if (preflight_counter >= (uint64_t(1) << 63))
            throw std::runtime_error("packed reservoir RNG counter is outside [0, 2**63)");

        Transaction *transaction = mutation_transaction();
        if (transaction != nullptr)
        {
            // Reserve the complete undo suffix before changing slots. A failed
            // allocation therefore leaves both the reservoir and journal intact.
            transaction->replacements.reserve(
                transaction->replacements.size() + output_count);
        }

        uint64_t output_position = 0;
        for (uint64_t position = 0; position < input_count; ++position)
        {
            const uint64_t item = input_view(position);
            if (slots_.size() < capacity_)
            {
                slots_.push_back(item);
                continue;
            }
            const uint64_t index = output_view(output_position);
            output_view(output_position++) = slots_[index];
            if (transaction != nullptr)
                transaction->replacements.push_back({index, slots_[index]});
            slots_[index] = slots_.back();
            slots_.back() = item;
        }
        rng_counter_ = preflight_counter;
        offered_ += input_count;
        emitted_ += output_count;
        if (slots_.size() > peak_occupancy_)
            peak_occupancy_ = slots_.size();
        return output;
    }

    py::array_t<uint64_t> drain(
        py::array_t<uint64_t, py::array::c_style> order)
    {
        if (closed_)
            return py::array_t<uint64_t>(0);
        if (order.ndim() != 1 || static_cast<size_t>(order.shape(0)) != slots_.size())
            throw std::invalid_argument("packed reservoir drain order has incorrect length");
        const auto order_view = order.unchecked<1>();
        std::vector<uint8_t> seen(slots_.size(), 0);
        py::array_t<uint64_t> output(slots_.size());
        auto output_view = output.mutable_unchecked<1>();
        for (size_t position = 0; position < slots_.size(); ++position)
        {
            const uint64_t index = order_view(position);
            if (index >= slots_.size() || seen[index])
                throw std::invalid_argument("packed reservoir drain order is not a permutation");
            seen[index] = 1;
            output_view(position) = slots_[index];
        }

        Transaction *transaction = mutation_transaction();
        if (transaction != nullptr)
        {
            // Draining is an epoch-boundary operation, so retain one full
            // pre-drain image only for the transaction that closes the core.
            // Copy before mutation to preserve strong exception safety.
            transaction->drain_slots.reset(new std::vector<uint64_t>(slots_));
        }
        emitted_ += slots_.size();
        slots_.clear();
        closed_ = true;
        return output;
    }

    uint64_t begin_transaction(size_t expected_replacements)
    {
        if (has_active_transaction())
            throw std::runtime_error("packed reservoir transaction is already active");
        if (next_transaction_id_ == std::numeric_limits<uint64_t>::max())
            throw std::runtime_error("packed reservoir transaction ID space is exhausted");
        const uint64_t transaction_id = next_transaction_id_;
        Transaction transaction(
            transaction_id,
            slots_.size(),
            rng_counter_,
            offered_,
            emitted_,
            peak_occupancy_,
            closed_);
        transaction.replacements.reserve(expected_replacements);
        transactions_.push_back(std::move(transaction));
        transactions_enabled_ = true;
        ++next_transaction_id_;
        return transaction_id;
    }

    void seal_transaction(uint64_t transaction_id)
    {
        if (!has_active_transaction() || transactions_.back().id != transaction_id)
            throw std::runtime_error("packed reservoir transaction is not active");
        transactions_.back().sealed = true;
    }

    void commit_transactions(const std::vector<uint64_t> &transaction_ids)
    {
        if (has_active_transaction())
            throw std::runtime_error("cannot commit while a reservoir transaction is active");
        if (transaction_ids.empty())
            throw std::invalid_argument("packed reservoir commit requires an endpoint");

        // Validate every token endpoint before releasing any undo state. An
        // endpoint may skip transactions absorbed by that token, such as the
        // terminal no-batch probe, but endpoint order must remain FIFO.
        auto transaction = transactions_.cbegin();
        for (const uint64_t transaction_id : transaction_ids)
        {
            while (transaction != transactions_.cend() &&
                   transaction->id < transaction_id)
                ++transaction;
            if (transaction == transactions_.cend() ||
                transaction->id != transaction_id)
                throw std::runtime_error("packed reservoir commit endpoint is not pending in FIFO order");
            ++transaction;
        }

        const uint64_t final_id = transaction_ids.back();
        do
        {
            transactions_.pop_front();
        } while (!transactions_.empty() && transactions_.front().id <= final_id);
    }

    void rollback_uncommitted()
    {
        for (auto transaction = transactions_.rbegin();
             transaction != transactions_.rend();
             ++transaction)
        {
            rollback_live_slots(slots_, *transaction);
            rng_counter_ = transaction->rng_counter_before;
            offered_ = transaction->offered_before;
            emitted_ = transaction->emitted_before;
            peak_occupancy_ = transaction->peak_occupancy_before;
            closed_ = transaction->closed_before;
        }
        transactions_.clear();
    }

    void rollback_transaction(uint64_t transaction_id)
    {
        if (transactions_.empty() || transactions_.back().id != transaction_id)
            throw std::runtime_error(
                "only the latest packed reservoir transaction can roll back");
        Transaction &transaction = transactions_.back();
        rollback_live_slots(slots_, transaction);
        rng_counter_ = transaction.rng_counter_before;
        offered_ = transaction.offered_before;
        emitted_ = transaction.emitted_before;
        peak_occupancy_ = transaction.peak_occupancy_before;
        closed_ = transaction.closed_before;
        transactions_.pop_back();
    }

    py::tuple committed_snapshot() const
    {
        std::vector<uint64_t> slots(slots_);
        uint64_t rng_counter = rng_counter_;
        uint64_t offered = offered_;
        uint64_t emitted = emitted_;
        uint64_t peak_occupancy = peak_occupancy_;
        bool closed = closed_;
        for (auto transaction = transactions_.rbegin();
             transaction != transactions_.rend();
             ++transaction)
        {
            rollback_copied_slots(slots, *transaction);
            rng_counter = transaction->rng_counter_before;
            offered = transaction->offered_before;
            emitted = transaction->emitted_before;
            peak_occupancy = transaction->peak_occupancy_before;
            closed = transaction->closed_before;
        }

        py::array_t<uint64_t> slot_array(slots.size());
        auto slot_view = slot_array.mutable_unchecked<1>();
        for (size_t index = 0; index < slots.size(); ++index)
            slot_view(index) = slots[index];
        return py::make_tuple(
            slot_array,
            rng_counter,
            offered,
            emitted,
            peak_occupancy,
            closed);
    }

    py::array_t<uint64_t> slots() const
    {
        py::array_t<uint64_t> output(slots_.size());
        auto view = output.mutable_unchecked<1>();
        for (size_t index = 0; index < slots_.size(); ++index)
            view(index) = slots_[index];
        return output;
    }

    void restore(
        py::array_t<uint64_t, py::array::c_style> slots,
        uint64_t rng_counter,
        uint64_t offered,
        uint64_t emitted,
        uint64_t peak_occupancy,
        bool closed)
    {
        if (slots.ndim() != 1 || static_cast<uint64_t>(slots.shape(0)) > capacity_)
            throw std::invalid_argument("packed reservoir restore slots exceed capacity");
        if (rng_counter >= (uint64_t(1) << 63))
            throw std::invalid_argument("packed reservoir RNG counter is outside [0, 2**63)");
        if (offered != emitted + static_cast<uint64_t>(slots.shape(0)))
            throw std::invalid_argument("packed reservoir restore violates exact-once accounting");
        if (peak_occupancy < static_cast<uint64_t>(slots.shape(0)) || peak_occupancy > capacity_)
            throw std::invalid_argument("packed reservoir restore peak occupancy is invalid");
        const auto view = slots.unchecked<1>();
        slots_.assign(view.data(0), view.data(0) + view.shape(0));
        rng_counter_ = rng_counter;
        offered_ = offered;
        emitted_ = emitted;
        peak_occupancy_ = peak_occupancy;
        closed_ = closed;
        transactions_.clear();
        transactions_enabled_ = false;
    }

    uint64_t occupancy() const { return slots_.size(); }
    uint64_t rng_counter() const { return rng_counter_; }
    uint64_t offered() const { return offered_; }
    uint64_t emitted() const { return emitted_; }
    uint64_t peak_occupancy() const { return peak_occupancy_; }
    bool closed() const { return closed_; }
    size_t pending_transaction_count() const { return transactions_.size(); }
    std::vector<uint64_t> pending_transaction_ids() const
    {
        std::vector<uint64_t> ids;
        ids.reserve(transactions_.size());
        for (const auto &transaction : transactions_)
            ids.push_back(transaction.id);
        return ids;
    }
    size_t pending_undo_entries() const
    {
        size_t count = 0;
        for (const auto &transaction : transactions_)
            count += transaction.replacements.size();
        return count;
    }
    size_t pending_journal_bytes() const
    {
        size_t bytes = transactions_.size() * sizeof(Transaction);
        for (const auto &transaction : transactions_)
        {
            bytes += transaction.replacements.capacity() * sizeof(ReplacementUndo);
            if (transaction.drain_slots)
                bytes += transaction.drain_slots->capacity() * sizeof(uint64_t);
        }
        return bytes;
    }

private:
    struct ReplacementUndo
    {
        uint64_t index;
        uint64_t old_value;
    };
    static_assert(
        sizeof(ReplacementUndo) == 2 * sizeof(uint64_t),
        "packed reservoir replacement undo must remain two uint64 values");

    struct Transaction
    {
        Transaction(
            uint64_t transaction_id,
            size_t occupancy,
            uint64_t rng_counter,
            uint64_t offered,
            uint64_t emitted,
            uint64_t peak_occupancy,
            bool closed)
            : id(transaction_id),
              occupancy_before(occupancy),
              rng_counter_before(rng_counter),
              offered_before(offered),
              emitted_before(emitted),
              peak_occupancy_before(peak_occupancy),
              closed_before(closed)
        {
        }

        uint64_t id;
        size_t occupancy_before;
        uint64_t rng_counter_before;
        uint64_t offered_before;
        uint64_t emitted_before;
        uint64_t peak_occupancy_before;
        bool closed_before;
        bool sealed = false;
        std::vector<ReplacementUndo> replacements;
        std::unique_ptr<std::vector<uint64_t>> drain_slots;
    };

    bool has_active_transaction() const
    {
        return !transactions_.empty() && !transactions_.back().sealed;
    }

    Transaction *mutation_transaction()
    {
        if (!transactions_enabled_)
            return nullptr;
        if (!has_active_transaction())
            throw std::runtime_error(
                "packed reservoir mutation requires an active transaction");
        return &transactions_.back();
    }

    static void rollback_replacements(
        std::vector<uint64_t> &slots,
        const Transaction &transaction)
    {
        for (auto replacement = transaction.replacements.rbegin();
             replacement != transaction.replacements.rend();
             ++replacement)
        {
            if (slots.empty() || replacement->index >= slots.size())
                throw std::runtime_error("packed reservoir undo journal is inconsistent");
            // At reverse time the replaced slot contains the old tail value.
            // Move it back to the tail, then restore the evicted slot value.
            slots.back() = slots[replacement->index];
            slots[replacement->index] = replacement->old_value;
        }
        if (slots.size() < transaction.occupancy_before)
            throw std::runtime_error("packed reservoir undo occupancy is inconsistent");
        slots.resize(transaction.occupancy_before);
    }

    static void rollback_live_slots(
        std::vector<uint64_t> &slots,
        Transaction &transaction)
    {
        // Swapping the retained pre-drain vector avoids allocation in the
        // destructive rollback path. Normal replacement rollback only shrinks.
        if (transaction.drain_slots)
            slots.swap(*transaction.drain_slots);
        rollback_replacements(slots, transaction);
    }

    static void rollback_copied_slots(
        std::vector<uint64_t> &slots,
        const Transaction &transaction)
    {
        if (transaction.drain_slots)
            slots = *transaction.drain_slots;
        rollback_replacements(slots, transaction);
    }

    uint64_t random_slot(uint64_t &counter, uint64_t upper) const
    {
        // Unsigned wraparound computes 2**64 modulo upper without relying on
        // a compiler-specific 128-bit integer type.
        const uint64_t remainder = (uint64_t(0) - upper) % upper;
        const uint64_t limit = uint64_t(0) - remainder;
        while (true)
        {
            if (counter >= (uint64_t(1) << 63))
                throw std::runtime_error("packed reservoir RNG counter is outside [0, 2**63)");
            uint64_t value = rng_base_ + counter++ * 0x9E3779B97F4A7C15ULL;
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL);
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EBULL);
            value ^= value >> 31;
            if (remainder == 0 || value < limit)
                return value % upper;
        }
    }

    uint64_t capacity_;
    uint64_t rng_base_;
    std::vector<uint64_t> slots_;
    uint64_t rng_counter_ = 0;
    uint64_t offered_ = 0;
    uint64_t emitted_ = 0;
    uint64_t peak_occupancy_ = 0;
    bool closed_ = false;
    std::deque<Transaction> transactions_;
    uint64_t next_transaction_id_ = 1;
    bool transactions_enabled_ = false;
};

PYBIND11_MODULE(dataset_planner_cpp, module)
{
    module.doc() = "Packed native dataset planner primitives";
    py::class_<PackedUInt64Reservoir>(module, "PackedUInt64Reservoir")
        .def(py::init<uint64_t, uint64_t>())
        .def("offer", &PackedUInt64Reservoir::offer)
        .def("drain", &PackedUInt64Reservoir::drain)
        .def("slots", &PackedUInt64Reservoir::slots)
        .def(
            "begin_transaction",
            &PackedUInt64Reservoir::begin_transaction,
            py::arg("expected_replacements") = 0)
        .def("seal_transaction", &PackedUInt64Reservoir::seal_transaction)
        .def("commit_transactions", &PackedUInt64Reservoir::commit_transactions)
        .def("rollback_transaction", &PackedUInt64Reservoir::rollback_transaction)
        .def("rollback_uncommitted", &PackedUInt64Reservoir::rollback_uncommitted)
        .def("committed_snapshot", &PackedUInt64Reservoir::committed_snapshot)
        .def("restore", &PackedUInt64Reservoir::restore)
        .def_property_readonly("occupancy", &PackedUInt64Reservoir::occupancy)
        .def_property_readonly("rng_counter", &PackedUInt64Reservoir::rng_counter)
        .def_property_readonly("offered", &PackedUInt64Reservoir::offered)
        .def_property_readonly("emitted", &PackedUInt64Reservoir::emitted)
        .def_property_readonly("peak_occupancy", &PackedUInt64Reservoir::peak_occupancy)
        .def_property_readonly("closed", &PackedUInt64Reservoir::closed)
        .def_property_readonly(
            "pending_transaction_count",
            &PackedUInt64Reservoir::pending_transaction_count)
        .def_property_readonly(
            "pending_transaction_ids",
            &PackedUInt64Reservoir::pending_transaction_ids)
        .def_property_readonly(
            "pending_undo_entries",
            &PackedUInt64Reservoir::pending_undo_entries)
        .def_property_readonly(
            "pending_journal_bytes",
            &PackedUInt64Reservoir::pending_journal_bytes);
}
