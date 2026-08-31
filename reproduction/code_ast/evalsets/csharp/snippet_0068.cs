using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;

namespace SimSharp.Visualization.Processor {
  public class JsonWriter : FramesProcessor {
    protected string target;
    protected StringWriter stringWriter;
    protected JsonTextWriter writer;
    protected string name;
    protected bool first = true;

    public JsonWriter() : this(Directory.GetParent(System.Environment.CurrentDirectory).Parent.FullName) { }

    public JsonWriter(string target) {
      this.target = target;
      this.stringWriter = new StringWriter();
      this.writer = new JsonTextWriter(stringWriter);
    }

    public void SendStart(AnimationConfig config) {
      this.name = config.Name;

      writer.WriteStartObject();
      config.WriteJson(writer);
      writer.WritePropertyName("frames");
      writer.WriteStartArray();
    }

    public void SendFrame(string frame) {
      if (!first)
        writer.WriteRaw(",");

      writer.WriteRaw(frame);
      first = false;
    }

    public virtual void SendStop() {
      writer.WriteEndArray();
      writer.WriteEndObject();

      Write();
    }

    private void Write() {
      File.WriteAllText(Path.Combine(target, Regex.Replace(name, @"\s+", "") + ".json"), stringWriter.ToString());
    }
  }
}