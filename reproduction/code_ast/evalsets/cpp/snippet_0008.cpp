#include <obstruction_free_dictionary.hpp>

namespace nonblocking
{

    template<> versioned<obstruction_free_dictionary::bucket_state>::versioned(unsigned int version, obstruction_free_dictionary::bucket_state value) : _version{version}
    {
        switch (value)
        {
            case obstruction_free_dictionary::bucket_state::empty: _value = 0; break;
            case obstruction_free_dictionary::bucket_state::busy: _value = 1; break;
            case obstruction_free_dictionary::bucket_state::inserting: _value = 2; break;
            case obstruction_free_dictionary::bucket_state::member: _value = 3; break;
            case obstruction_free_dictionary::bucket_state::updating: _value = 4; break;
        }
    }

    template <> versioned<obstruction_free_dictionary::bucket_state>::versioned() noexcept : _version{0}, _value{0} {}

    template <> bool versioned<obstruction_free_dictionary::bucket_state>::operator==(versioned<obstruction_free_dictionary::bucket_state> other)
    {
        return (_version == other._version) && (_value == other._value);
    }

    template<> obstruction_free_dictionary::bucket_state versioned<obstruction_free_dictionary::bucket_state>::value()
    {
        switch (_value)
        {
            case 0: return obstruction_free_dictionary::bucket_state::empty;
            case 1: return obstruction_free_dictionary::bucket_state::busy;
            case 2: return obstruction_free_dictionary::bucket_state::inserting;
            case 3: return obstruction_free_dictionary::bucket_state::member;
            case 4: return obstruction_free_dictionary::bucket_state::updating;
            default: return obstruction_free_dictionary::bucket_state::busy;
        }
    }

    template<> unsigned int versioned<obstruction_free_dictionary::bucket_state>::version()
    {
        return _version;
    }

    obstruction_free_dictionary::bucket::bucket(unsigned int key, unsigned int value, unsigned int version, bucket_state state) : key{key}, state(versioned<obstruction_free_dictionary::bucket_state>(version, state)), value(versioned<unsigned int>(version, value)), copy(versioned<unsigned int>(version, 0)) {}

    obstruction_free_dictionary::bucket::bucket() : key{0}, state(versioned<obstruction_free_dictionary::bucket_state>()), value(versioned<unsigned int>()), copy(versioned<unsigned int>()) {}

    obstruction_free_dictionary::bucket::bucket(unsigned int key, versioned<unsigned int> value, versioned<unsigned int> copy, versioned<bucket_state> state) : key{key}, value{value}, copy{copy}, state{state} {}
    obstruction_free_dictionary::bucket* obstruction_free_dictionary::bucket_at(unsigned int h, int index)
    {
        return &buckets[(h+index*(index+1)/2)%size];
    }

    obstruction_free_dictionary::bucket::bucket(const obstruction_free_dictionary::bucket& b)
    {
        key.store(b.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
        state.store(b.state.load(std::memory_order_relaxed), std::memory_order_relaxed);
        value.store(b.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
        copy.store(b.copy.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    bool obstruction_free_dictionary::does_bucket_contain_collision(unsigned int h, int index)
    {
        versioned<bucket_state> state;
        versioned<bucket_state> later_state;
        unsigned int key;
        do
        {
            bucket* bkt_at = bucket_at(h, index);
            state = bkt_at->state.load(std::memory_order_acquire);
            key = bkt_at->key.load(std::memory_order_acquire);
            later_state = bkt_at->state.load(std::memory_order_relaxed);
        } while (!(state == later_state));
        if ((state.value() == inserting) || (state.value() == member) || (state.value() == updating))
        {
            if (hash(key) == h)
            {
                return true;
            }
        }
        return false;
    }

    obstruction_free_dictionary::obstruction_free_dictionary(unsigned int size) : base(size)
    {
        buckets = new bucket[size];
    }

    obstruction_free_dictionary::~obstruction_free_dictionary()
    {
        delete buckets;
    }

    std::pair<obstruction_free_dictionary::bucket*, obstruction_free_dictionary::bucket> obstruction_free_dictionary::read_bucket(unsigned int key)
    {
        unsigned int h = hash(key);
        auto max = get_probe_bound(h);
        for (unsigned int i = 0; i <= max; ++i)
        {
            versioned<bucket_state> state;
            versioned<bucket_state> later_state;
            versioned<unsigned int> value;
            versioned<unsigned int> copy;
            unsigned int bkt_key;
            do
            {
                state = bucket_at(h, i)->state.load(std::memory_order_acquire);
                bkt_key = bucket_at(h, i)->key.load(std::memory_order_acquire);
                if (bkt_key != key)
                {
                    break;
                }
                if ((state.value() == busy) || (state.value() == inserting) || (state.value() == empty))
                {
                    break;
                }
                value = bucket_at(h, i)->value.load(std::memory_order_acquire);
                copy = bucket_at(h, i)->copy.load(std::memory_order_acquire);
                later_state = bucket_at(h, i)->state.load(std::memory_order_relaxed);
            }
            while (!(state == later_state));
            if (bkt_key == key)
            {
                if ((state.value() == member) || (state.value() == updating))
                {
                    return std::make_pair(bucket_at(h, i), bucket(key, value, copy, state));
                }
            }
        }
        return std::make_pair(nullptr, bucket());
    }

    std::pair<bool, unsigned int> obstruction_free_dictionary::lookup(unsigned int key)
    {
        std::pair<bucket*, bucket> bkt = read_bucket(key);
        if (std::get<0>(bkt))
        {
            auto bkt_at = std::get<1>(bkt);
            if (bkt_at.state.load(std::memory_order_relaxed).value() == member)
            {
                return std::make_pair(true, bkt_at.value.load(std::memory_order_relaxed).value());
            }
            else if (bkt_at.state.load(std::memory_order_relaxed).value() == updating)
            {
                return std::make_pair(true, bkt_at.copy.load(std::memory_order_relaxed).value());
            }
        }
        return std::make_pair(false, 0);
    }

    bool obstruction_free_dictionary::erase(unsigned int key)
    {
        unsigned int h = hash(key);
        auto max = get_probe_bound(h);
        for (unsigned int i = 0; i <= max; ++i)
        {
            while (true)
            {
                auto* bkt_at = bucket_at(h, i);
                auto state = bkt_at->state.load(std::memory_order_acquire);
                auto bkt_key = bkt_at->key.load(std::memory_order_acquire);
                if (((state.value() == member) || (state.value() == updating)) && (bkt_key == key))
                {
                    if (std::atomic_compare_exchange_strong_explicit(&bkt_at->state, &state, versioned<bucket_state>(state.version(), busy), std::memory_order_release, std::memory_order_acquire))
                    {
                        conditionally_lower_bound(h, i);
                        bkt_at->state.store(versioned<bucket_state>(state.version()+1, empty), std::memory_order_release);
                        return true;
                    }
                    // Some other operation (delete/update) got in our way -- try again
                    continue;
                }
                // No match: try another bucket
                break;
            }
        }
        return false;
    }

    void obstruction_free_dictionary::set(unsigned int key, unsigned int value)
    {
        while (true)
        {
            bool aborted = false;
            std::pair<bucket*, bucket> bkt = read_bucket(key);
            if (auto* bkt_at = std::get<0>(bkt))
            {
                // Update
                auto bkt_value = std::get<1>(bkt);
                auto state = bkt_value.state.load(std::memory_order_relaxed);
                if (state.value() == member)
                {
                    // Copy value to copy
                    auto old_copy = bkt_value.copy.load();
                    if (!std::atomic_compare_exchange_strong_explicit(&bkt_at->copy, &old_copy, bkt_value.value.load(std::memory_order_relaxed), std::memory_order_release, std::memory_order_relaxed))
                    {
                        // Failed to copy -- another updater is present: abort
                        continue;
                    }
                }
                auto new_state = versioned<bucket_state>(state.version() + 1, updating);
                if (!std::atomic_compare_exchange_strong_explicit(&bkt_at->state, &state, new_state, std::memory_order_release, std::memory_order_relaxed))
                {
                    // Status modified: abort
                    continue;
                }
                // Ensure that ALL the previous happened before modifying the value
                std::atomic_thread_fence(std::memory_order_acquire);
                // Update the value
                auto old_value = bkt_value.value.load(std::memory_order_relaxed);
                if (!std::atomic_compare_exchange_strong_explicit(&bkt_at->value, &old_value, versioned<unsigned int>(state.version() + 1, value), std::memory_order_relaxed, std::memory_order_relaxed))
                {
                    // Value modified: abort
                    continue;
                }
                // Move back to member status
                if (std::atomic_compare_exchange_strong_explicit(&bkt_at->state, &new_state, versioned<bucket_state>(state.version() + 1, member), std::memory_order_release, std::memory_order_relaxed))
                {
                    return;
                }
            }
            else
            {
                // Insert
                unsigned int h = hash(key);
                int i = -1;
                unsigned int version;
                versioned<bucket_state> old_state;
                versioned<bucket_state> new_state;
                do
                {
                    if (++i > size)
                    {
                        throw "Table full";
                    }
                    version = bucket_at(h, i)->state.load(std::memory_order_acquire).version();
                    old_state = versioned<bucket_state>(version, empty);
                    new_state = versioned<bucket_state>(version, busy);
                }
                while (!std::atomic_compare_exchange_strong_explicit(&bucket_at(h, i)->state, &old_state, new_state, std::memory_order_release, std::memory_order_relaxed));
                unsigned int new_version = version+1;
                bucket_at(h, i)->key.store(key, std::memory_order_relaxed);
                bucket_at(h, i)->value.store(versioned<unsigned int>(new_version, value), std::memory_order_relaxed);
                versioned<bucket_state> inserting_state;
                do
                {
                    inserting_state = versioned<bucket_state>(new_version, inserting);
                    bucket_at(h, i)->state.store(inserting_state, std::memory_order_release);
                    conditionally_raise_bound(h, i);
                    auto max = get_probe_bound(h);
                    for (unsigned int j = 0; j <= max; ++j)
                    {
                        if (j != i)
                        {
                            auto* bkt_at = bucket_at(h, j);
                            unsigned int bkt_key;
                            do
                            {
                                old_state = bkt_at->state.load(std::memory_order_acquire);
                                bkt_key = bkt_at->key.load(std::memory_order_acquire);
                                if ((bkt_key != key) || (old_state.value() == busy) || (old_state.value() == empty))
                                {
                                    break;
                                }
                                new_state = bkt_at->state.load(std::memory_order_relaxed);
                            }
                            while (!(old_state == new_state));
                            if (bkt_key == key)
                            {
                                if (old_state.value() == inserting)
                                {
                                    auto j_insert = versioned<bucket_state>(old_state.version(), inserting);
                                    std::atomic_compare_exchange_strong_explicit(&bkt_at->state, &j_insert, versioned<bucket_state>(old_state.version(), busy), std::memory_order_release, std::memory_order_relaxed);
                                }
                                else if ((old_state.value() == member) || (old_state.value() == updating))
                                {
                                    bucket_at(h, i)->state.store(versioned<bucket_state>(version, busy));
                                    conditionally_lower_bound(h, i);
                                    bucket_at(h, i)->state.store(versioned<bucket_state>(version, empty));
                                    aborted = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (aborted)
                    {
                        break;
                    }
                }
                while (!std::atomic_compare_exchange_strong_explicit(&bucket_at(h, i)->state, &inserting_state, versioned<bucket_state>(new_version, member), std::memory_order_release, std::memory_order_relaxed));
                if (!aborted)
                {
                    return;
                }
            }
        }
    }
}