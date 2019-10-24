/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <miopen/errors.hpp>
#include <miopen/type_name.hpp>

#include <memory>
#include <typeinfo>

namespace miopen {
struct ConvolutionContext;

// Marker for explicit empty initialization.
struct InializeAsEmpty final
{
};

namespace solver {
struct AnyPerformanceConfig final
{
    private:
    struct PerformanceConfigConcept
    {
        virtual ~PerformanceConfigConcept()                                  = default;
        virtual bool SetNextValue()                                          = 0;
        virtual bool IsValid(const ConvolutionContext& ctx) const            = 0;
        virtual bool operator==(const PerformanceConfigConcept& other) const = 0;
        virtual bool IsOfType(const std::type_info& type) const              = 0;
        virtual void Serialize(std::ostream& stream) const                   = 0;
        virtual bool Deserialize(const std::string& s)                       = 0;
        virtual std::unique_ptr<PerformanceConfigConcept> Clone() const      = 0;
    };

    template <class TPerformanceConfig>
    struct PerformanceConfigModelBase : public PerformanceConfigConcept
    {
        void Serialize(std::ostream& stream) const final { config.Serialize(stream); }
        bool Deserialize(const std::string& s) final { return config.Deserialize(s); }

        bool IsOfType(const std::type_info& type) const final
        {
            return type == typeid(TPerformanceConfig);
        }

        std::unique_ptr<PerformanceConfigConcept> Clone() const final
        {
            return std::make_unique<PerformanceConfigModel<TPerformanceConfig>>(
                TPerformanceConfig{config});
        }

        TPerformanceConfig& Get() { return config; }
        const TPerformanceConfig& Get() const { return config; }

        protected:
        TPerformanceConfig config;

        PerformanceConfigModelBase(TPerformanceConfig&& config_) : config(std::move(config_)) {}
    };

    template <class TPerformanceConfig, class = bool>
    struct PerformanceConfigModel final : public PerformanceConfigModelBase<TPerformanceConfig>
    {
        PerformanceConfigModel(TPerformanceConfig&& config_)
            : PerformanceConfigModelBase<TPerformanceConfig>(std::move(config_))
        {
        }

        bool SetNextValue() final
        {
            MIOPEN_THROW("Using unimplemented generic search related methods on " +
                         get_type_name<TPerformanceConfig>());
        }

        bool IsValid(const ConvolutionContext&) const final
        {
            MIOPEN_THROW("Using unimplemented generic search related methods on " +
                         get_type_name<TPerformanceConfig>());
        }

        bool operator==(const PerformanceConfigConcept&) const final
        {
            MIOPEN_THROW("Using unimplemented generic search related methods on " +
                         get_type_name<TPerformanceConfig>());
        }
    };

    template <class TPerformanceConfig>
    struct PerformanceConfigModel<TPerformanceConfig,
                                  decltype(std::declval<TPerformanceConfig>().SetNextValue())>
        final : public PerformanceConfigModelBase<TPerformanceConfig>
    {
        PerformanceConfigModel(TPerformanceConfig&& config_)
            : PerformanceConfigModelBase<TPerformanceConfig>(std::move(config_))
        {
        }

        bool SetNextValue() final { return this->config.SetNextValue(); }
        bool IsValid(const ConvolutionContext& ctx) const final
        {
            return this->config.IsValid(ctx);
        }

        bool operator==(const PerformanceConfigConcept& other) const final
        {
            const auto& casted_other =
                dynamic_cast<const PerformanceConfigModel<TPerformanceConfig>&>(other);
            return this->config == casted_other.config;
        }
    };

    public:
    AnyPerformanceConfig(InializeAsEmpty&&) : config(nullptr) {}

    AnyPerformanceConfig(const AnyPerformanceConfig& other)
        : config(other.IsEmpty() ? nullptr : other.config->Clone())
    {
    }

    AnyPerformanceConfig(AnyPerformanceConfig&& other) noexcept : config(std::move(other.config)) {}

    template <class TPerformanceConfig>
    AnyPerformanceConfig(TPerformanceConfig other)
        : config(std::make_unique<PerformanceConfigModel<TPerformanceConfig>>(std::move(other)))
    {
    }

    bool IsEmpty() const { return config == nullptr; }

    bool SetNextValue()
    {
        CheckIfEmpty();
        return config->SetNextValue();
    }

    bool IsValid(const ConvolutionContext& ctx) const
    {
        CheckIfEmpty();
        return config->IsValid(ctx);
    }

    bool operator==(const AnyPerformanceConfig& other) const
    {
        CheckIfEmpty();
        return *config == *other.config;
    }

    bool IsOfType(const std::type_info& type) const
    {
        CheckIfEmpty();
        return config->IsOfType(type);
    }

    void Serialize(std::ostream& stream) const
    {
        CheckIfEmpty();
        config->Serialize(stream);
    }

    bool Deserialize(const std::string& s)
    {
        CheckIfEmpty();
        return config->Deserialize(s);
    }

    template <class TPerformanceConfig>
    bool IsOfType() const
    {
        return IsOfType(typeid(TPerformanceConfig));
    }

    friend std::ostream& operator<<(std::ostream& os, const AnyPerformanceConfig& c)
    {
        c.config->Serialize(os);
        return os;
    }

    void Swap(AnyPerformanceConfig& other) { std::swap(config, other.config); }

    AnyPerformanceConfig& operator=(AnyPerformanceConfig&& other) noexcept
    {
        if(this == &other)
            return *this;
        config = std::move(other.config);
        return *this;
    }

    AnyPerformanceConfig& operator=(const AnyPerformanceConfig& other)
    {
        if(this == &other)
            return *this;
        config = other.IsEmpty() ? nullptr : other.config->Clone();
        return *this;
    }

    template <class TPerformanceConfig>
    TPerformanceConfig& CastTo()
    {
        CheckIfEmpty();
        if(!IsOfType<TPerformanceConfig>())
            MIOPEN_THROW("Invalid AnyPerformanceConfig cast: config type doesn't match.");

        auto& casted_config = dynamic_cast<PerformanceConfigModel<TPerformanceConfig>&>(*config);
        return casted_config.Get();
    }

    template <class TPerformanceConfig>
    const TPerformanceConfig& CastTo() const
    {
        CheckIfEmpty();
        if(!IsOfType<TPerformanceConfig>())
            MIOPEN_THROW("Invalid AnyPerformanceConfig cast: config type doesn't match.");

        const auto& casted_config =
            dynamic_cast<const PerformanceConfigModel<TPerformanceConfig>&>(*config);
        return casted_config.Get();
    }

    private:
    std::unique_ptr<PerformanceConfigConcept> config;

    void CheckIfEmpty() const
    {
        if(IsEmpty())
            MIOPEN_THROW("Using config methods on an empty AnyPerformanceConfig.");
    }
};

inline void swap(AnyPerformanceConfig& left, AnyPerformanceConfig& right) { left.Swap(right); }

} // namespace solver
} // namespace miopen
