// Copyright 2021-2022 <NAME>
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace AInq.Optional;

/// <summary> Value-or-error type </summary>
/// <typeparam name="T"> Value type </typeparam>
public abstract class Try<T> : IEquatable<Try<T>>, IEquatable<T>
{
    /// <summary> Check if item is success </summary>
    public bool Success => IsSuccess();

    /// <summary> Item value (if success) </summary>
    public T Value => IsSuccess() ? GetValue() : throw GetError();

    /// <summary> Exception or null if success </summary>
    public Exception? Error => IsSuccess() ? null : GetError();

    /// <inheritdoc />
    public bool Equals(T? other)
        => Success && EqualityComparer<T?>.Default.Equals(Value, other);

    /// <inheritdoc />
    public bool Equals(Try<T>? other)
        => other is not null && Success && other.Success && EqualityComparer<T>.Default.Equals(Value, other.Value);

    private protected abstract bool IsSuccess();
    private protected abstract T GetValue();
    private protected abstract Exception GetError();

    /// <summary> Throw if contains exception </summary>
    public Try<T> Throw()
    {
        if (!IsSuccess()) throw GetError();
        return this;
    }

    // <summary> Throw if contains exception of target type </summary>
    /// <param name="exceptionType"> Target exception type </param>
    public Try<T> Throw(Type exceptionType)
    {
        if (IsSuccess()) return this;
        var exception = GetError();
        if (exception.GetType() == exceptionType) throw exception;
        return this;
    }

    /// <summary> Throw if contains exception of target type </summary>
    /// <typeparam name="TException"> Target exception type </typeparam>
    public Try<T> Throw<TException>()
        where TException : Exception
    {
        if (!IsSuccess() && GetError() is TException exception) throw exception;
        return this;
    }

    /// <inheritdoc />
    public override string ToString()
        => Success ? Value?.ToString() ?? "Null" : Error!.ToString();

    /// <inheritdoc />
    public override int GetHashCode()
        => Success ? Value?.GetHashCode() ?? 0 : 1;

    /// <inheritdoc />
    public override bool Equals(object? obj)
        => obj switch
        {
            Try<T> other => Equals(other),
            T value => Equals(value),
            _ => false
        };

    /// <summary> Equality comparison </summary>
    /// <param name="a"> First element </param>
    /// <param name="b"> Second element </param>
    public static bool operator ==(Try<T>? a, Try<T>? b)
        => a?.Equals(b) ?? b is null;

    /// <summary> Inequality comparison </summary>
    /// <param name="a"> First element </param>
    /// <param name="b"> Second element </param>
    public static bool operator !=(Try<T>? a, Try<T>? b)
        => !(a == b);

    /// <summary> Equality comparison </summary>
    /// <param name="a"> First element </param>
    /// <param name="b"> Second element </param>
    public static bool operator ==(Try<T>? a, T? b)
        => a?.Equals(b) ?? false;

    /// <summary> Inequality comparison </summary>
    /// <param name="a"> First element </param>
    /// <param name="b"> Second element </param>
    public static bool operator !=(Try<T>? a, T? b)
        => !(a == b);

    /// <summary> Equality comparison </summary>
    /// <param name="a"> First element </param>
    /// <param name="b"> Second element </param>
    public static bool operator ==(T? a, Try<T>? b)
        => b?.Equals(a) ?? false;

    /// <summary> Inequality comparison </summary>
    /// <param name="a"> First element </param>
    /// <param name="b"> Second element </param>
    public static bool operator !=(T? a, Try<T>? b)
        => !(a == b);

    // <summary> Create Try from value </summary>
    /// <param name="value"> Value </param>
    public static Try<T> FromValue(T value)
        => new TryValue(value);

    /// <summary> Create Try from exception </summary>
    /// <param name="exception"> Exception </param>
    public static Try<T> FromError(Exception exception)
        => new TryError(exception is AggregateException {InnerExceptions.Count: 1} aggregate ? aggregate.InnerExceptions[0] : exception);

    private sealed class TryValue : Try<T>
    {
        private readonly T _value;

        internal TryValue(T value)
            => _value = value;

        private protected override bool IsSuccess()
            => true;

        private protected override T GetValue()
            => _value;

        private protected override Exception GetError()
            => new();
    }

    private sealed class TryError : Try<T>
    {
        private readonly Exception _error;

        internal TryError(Exception error)
            => _error = error;

        private protected override bool IsSuccess()
            => false;

        private protected override T GetValue()
            => throw _error;

        private protected override Exception GetError()
            => _error;
    }
}