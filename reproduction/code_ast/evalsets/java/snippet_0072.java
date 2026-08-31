/**
 * MIT License
 *
 * Copyright (c) 2019 <NAME> and xjikll
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
 */
package org.tonygatins.tonysmith.xjikll.plant.graphics.drawable;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * 田块按钮的背景实现。
 * @author <NAME>
 */
public final class FieldBackgroundDrawable extends BackgroundDrawable
{
	//错误大小画笔
	private static final Paint ERROR_SIZE_PAINT =new Paint();
	
	/**
	 * 默认背景画笔。
	 */
	public static final Paint DEFAULT_PAINT = new Paint();
	
	//图片的大小
	private final Rect src;

	//不同状态时的不同画笔。
	private final Map<int[],Paint> statePaint = new HashMap<>();
	
	//初始条件列表
	private final Map<Integer,Paint> initStates = new HashMap<>();
	
	//当前画笔
	private Paint pen;
	
	//前次状态
	private int[] lastStates;

	/**
	 * 用一个bitmap构造对象。
	 * @param resource bitmap对象。
	 */
	public FieldBackgroundDrawable(Bitmap resource)
	{
		super(resource);
		src = new Rect(0,0,get().getWidth(),get().getHeight());
		pen=DEFAULT_PAINT;
	}

	/**
	 * 获得背景图片的大小。
	 * @return 代表图片大小的矩形对象。
	 */
	public Rect getRect()
	{
		return src;
	}

	/**
	 * {@hide}
	 * 添加状态的方法，不同状态的不同画笔。
	 * @param state 状态ID。
	 * @param paint 画笔对象。
	 */
	public void addStates(int state,Paint paint)
	{
		initStates.put(state,paint);
	}
	
	//内部添加状态
	private void addStates(int[] state,Paint paint)
	{
		statePaint.put(state,paint);
	}
	
	/**
	 * 特殊的绘制背景方法。
	 * @param canvas 系统提供的画布对象。
	 */
	@Override
	public void draw(Canvas canvas)
	{
		// TODO: Implement this method
		if(getBounds().centerX() == src.centerX() + 1 && getBounds().centerY() == src.centerY() + 1)
		{
			canvas.drawPaint(pen);
			canvas.drawBitmap(get(),null,new Rect(1,1,getBounds().width() - 1,getBounds().height() - 1),null);
		}
		else
		{
			canvas.drawPaint(ERROR_SIZE_PAINT);
			return;
		}
	}
	
	/**
	 * 状态改变时的处理方法。
	 * @param state 代表状态id的数组对象。
	 * @return 是否改变了状态(只有true)。
	 */
	@Override
	protected boolean onStateChange(int[] state)
	{
		// TODO: Implement this method
		Arrays.sort(state);
		if(Arrays.equals(state,lastStates))
		{
			return false;
		}
		
		Paint temp=statePaint.get(state);
		
		if(temp!=null)
		{
			pen=temp;
			lastStates=state;
			invalidateSelf();
			return true;
		}
		else
		{
			int index=Arrays.binarySearch(state,android.R.attr.state_pressed);
			if(index > -1)
			{
				addStates(state,initStates.get(android.R.attr.state_pressed));
				pen=initStates.get(android.R.attr.state_pressed);
			}
			else
			{
				addStates(state,DEFAULT_PAINT);
				pen=DEFAULT_PAINT;
			}
			lastStates=state;
			invalidateSelf();
			return true;
		}
	}
	
	//初始化默认画笔
	static
	{
		DEFAULT_PAINT.reset();
		DEFAULT_PAINT.setAlpha(255);
		DEFAULT_PAINT.setAntiAlias(true);
		DEFAULT_PAINT.setColor(Color.BLACK);
		
		ERROR_SIZE_PAINT.reset();
		ERROR_SIZE_PAINT.setAlpha(255);
		ERROR_SIZE_PAINT.setAntiAlias(true);
		ERROR_SIZE_PAINT.setColor(Color.RED);
	}

	/**
	 * 返回是否可变。
	 * @return 永远为true
	 */
	@Override
	public boolean isStateful()
	{
		// TODO: Implement this method
		return true;
	}
	
	/**
	 * 共享信息构造方法。
	 * @param another 另一个drawable。
	 */
	public FieldBackgroundDrawable(FieldBackgroundDrawable another)
	{
		super(another);
		src=another.src;
		initStates.putAll(another.initStates);
	}
}