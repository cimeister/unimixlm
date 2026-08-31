package com.projectkroma.terrabyte.renderer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.lwjgl.opengl.GL11;

import com.projectkroma.terrabyte.opengl.Texture;
import com.projectkroma.terrabyte.opengl.VAO;
import com.projectkroma.terrabyte.opengl.shader.Shader;

public class StandardShaderPipeline implements IShaderPipeline
{

	public Shader shader;

	private boolean autoClear;

	private List<PipelineObject> pipelineObjects;
	private Map<String, Object> shaderUniforms;

	public StandardShaderPipeline(Shader shader)
	{
		this.shader = shader;
		this.pipelineObjects = new ArrayList<PipelineObject>();
		this.shaderUniforms = new HashMap<String, Object>();
	}

	@Override
	public void setShaderUniform(String uniform, Object object)
	{
		this.shaderUniforms.put(uniform, object);
	}

	@Override
	public void remShaderUniform(String uniform)
	{
		this.shaderUniforms.remove(uniform);
	}

	public void addPipelineObject(PipelineObject pipelineObject)
	{
		this.pipelineObjects.add(pipelineObject);
	}

	@Override
	public void draw()
	{
		shader.bind();

		for (String uniform : shaderUniforms.keySet())
		{
			shader.load(uniform, shaderUniforms.get(uniform));
		}

		VAO lastVAO = null;
		for (PipelineObject po : pipelineObjects)
		{
			if (po.vao != lastVAO)
				po.vao.bind();
			Texture.unbind();
			if (po.texture != null)
			{
				po.texture.bind();
			}

			for (String uniform : po)
			{
				shader.load(uniform, po.getObjectUniform(uniform));
			}

			po.vao.drawArrays(GL11.GL_TRIANGLES);

			lastVAO = po.vao;
		}

		if (autoClear)
			pipelineObjects.clear();

		VAO.unbind();
		Texture.unbind();
		Shader.unbind();
	}

	@Override
	public void setAutoClear(boolean clear)
	{
		autoClear = clear;
	}

	@Override
	public boolean isAutoClear()
	{
		return autoClear;
	}
}