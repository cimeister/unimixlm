using GameEstate.Graphics;
using GameEstate.Graphics.OpenGL;

namespace GameEstate
{
    public interface IOpenGLGraphic : IEstateGraphic<object, Material, int, Shader>
    {
        // cache
        public GpuMeshBufferCache MeshBufferCache { get; }
        public QuadIndexBuffer QuadIndices { get; }
    }
}