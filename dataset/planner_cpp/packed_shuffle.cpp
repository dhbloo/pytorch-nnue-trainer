#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
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
            random_slot(preflight_counter, capacity_);
        if (preflight_counter >= (uint64_t(1) << 63))
            throw std::runtime_error("packed reservoir RNG counter is outside [0, 2**63)");

        uint64_t output_position = 0;
        for (uint64_t position = 0; position < input_count; ++position)
        {
            const uint64_t item = input_view(position);
            if (slots_.size() < capacity_)
            {
                slots_.push_back(item);
                continue;
            }
            const uint64_t index = random_slot(rng_counter_, capacity_);
            output_view(output_position++) = slots_[index];
            slots_[index] = slots_.back();
            slots_.back() = item;
        }
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
        emitted_ += slots_.size();
        slots_.clear();
        closed_ = true;
        return output;
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
    }

    uint64_t occupancy() const { return slots_.size(); }
    uint64_t rng_counter() const { return rng_counter_; }
    uint64_t offered() const { return offered_; }
    uint64_t emitted() const { return emitted_; }
    uint64_t peak_occupancy() const { return peak_occupancy_; }
    bool closed() const { return closed_; }

private:
    uint64_t random_slot(uint64_t &counter, uint64_t upper) const
    {
        const uint64_t remainder = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(1) << 64) % upper);
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
};

PYBIND11_MODULE(dataset_planner_cpp, module)
{
    module.doc() = "Packed native dataset planner primitives";
    py::class_<PackedUInt64Reservoir>(module, "PackedUInt64Reservoir")
        .def(py::init<uint64_t, uint64_t>())
        .def("offer", &PackedUInt64Reservoir::offer)
        .def("drain", &PackedUInt64Reservoir::drain)
        .def("slots", &PackedUInt64Reservoir::slots)
        .def("restore", &PackedUInt64Reservoir::restore)
        .def_property_readonly("occupancy", &PackedUInt64Reservoir::occupancy)
        .def_property_readonly("rng_counter", &PackedUInt64Reservoir::rng_counter)
        .def_property_readonly("offered", &PackedUInt64Reservoir::offered)
        .def_property_readonly("emitted", &PackedUInt64Reservoir::emitted)
        .def_property_readonly("peak_occupancy", &PackedUInt64Reservoir::peak_occupancy)
        .def_property_readonly("closed", &PackedUInt64Reservoir::closed);
}
