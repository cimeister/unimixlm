namespace Larva.Core.Serialization.Xml
{
    /// <summary>
    /// Xml序列化模块
    /// </summary>
    public static class XmlSerializationModule
    {
        /// <summary>
        /// 模块名
        /// </summary>
        public const string MODULE_NAME = "Larva.Core.Serialization.Xml";

        /// <summary>
        /// 使用Xml序列化
        /// </summary>
        /// <param name="manager">模块管理器</param>
        /// <param name="serializer">文本序列化器</param>
        /// <param name="canOverride">是否覆盖</param>
        /// <returns></returns>
        public static IModuleManager UseXmlSerialization(this IModuleManager manager, IXmlSerializer serializer, bool canOverride = false)
        {
            manager.Register(MODULE_NAME, serializer, canOverride);
            return manager;
        }

        /// <summary>
        /// 获取Xml序列化
        /// </summary>
        /// <param name="manager">模块管理器</param>
        /// <returns></returns>
        public static IXmlSerializer GetXmlSerializer(this IModuleManager manager)
        {
            return (IXmlSerializer)manager.Get(MODULE_NAME);
        }

        static XmlSerializationModule() { }

        /// <summary>
        /// 实例
        /// </summary>
        public static IXmlSerializer Instance
        {
            get { return ModuleManager.Instance.GetXmlSerializer(); }
        }
    }
}