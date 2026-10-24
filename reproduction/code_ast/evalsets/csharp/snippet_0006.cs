using System;

namespace Argo
{
    public interface IPacket
    {
        int Command { get; }

        int Sequence { get; }

        Span<byte> Body { get; }
    }
}