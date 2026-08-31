
using System;
using Core.KeyValues;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Microsoft.Extensions.DependencyInjection
{
    public static class InMemoryKeyValueExtensions
    {
        public static IKeyValueBuilder AddInMemory(this IKeyValueBuilder builder)
        {
            builder.Services.TryAddSingleton<IKeyValueStore, InMemoryKeyValueStore>();
            return builder;
        }
    }
}