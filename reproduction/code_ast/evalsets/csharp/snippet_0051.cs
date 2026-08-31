using System;
using System.Collections.Generic;

public class Box
{
	public int id;

	public string name;

	public string description;

	public Dictionary<int, int> items = new Dictionary<int, int>();

	public Dictionary<int, int> rate = new Dictionary<int, int>();

	public Dictionary<int, int> equips = new Dictionary<int, int>();

	public Dictionary<int, int> equipRate = new Dictionary<int, int>();
}