package edu.iastate.cs228.hw5;

import java.util.HashSet;
import java.util.ArrayList;
import java.util.Iterator;

public class Graph
{
  public class Edge {
    protected int weight;
    protected Vertex from;
    protected Vertex to;

    protected Edge(Vertex from, Vertex to, int weight)
    {
      this.from = from;
      this.to = to;
      this.weight = weight;
    }

    public int get_weight()
    {
      return weight;
    }

    public int get_from()
    {
      return from.index;
    }

    public int get_to()
    {
      return to.index;
    }
  }

  public class Vertex {
    protected int index;
    protected HashSet<Edge> edges;

    protected Vertex(int index)
    {
      this.index = index;
      edges = new HashSet<Edge>();
    }


    public int get_index()
    {
      return index;
    }

    public String toString()
    {
      String s;

      s = "";
      for (Edge e : edges) {
        s += e.to.index + "," + e.weight + " ";
      }

      return s;
    }

    public Iterable<Edge> get_edges()
    {
      return edges;
    }

    public int hashCode()
    {
      int tmp = index;
      if ( tmp < 0 ) tmp = - tmp;
      return tmp;
    }

    public boolean equals(Object obj)
    {
      if ( this == obj ) return true;
      if ( (obj == null) || (obj.getClass() != this.getClass()) )
           return false;
      // object must be Vertex at this point
      Vertex test = (Vertex) obj;
      return index == test.index;
    }

  };

  private ArrayList<Vertex> vertices;
  private int array_size = 128;

  public Graph()
  {
    int i;

    vertices = new ArrayList<Vertex>(array_size);

    for (i = 0; i < array_size; i++) {
      vertices.add(null);
    }
  }

  private void check_size(int size)
  {
    int old_size = array_size;

    while (size >= array_size) {
      array_size *= 2;
    }

    while (old_size != array_size) {
      vertices.add(null);
      old_size++;
    }
  }

  public void add_edge(int from, int to, int weight)
  {
    check_size(java.lang.Math.max(from, to));

    if (vertices.get(from) == null) {
      vertices.set(from, new Vertex(from));
    }
    if (vertices.get(to) == null) {
      vertices.set(to, new Vertex(to));
    }
    vertices.get(from).edges.add(new Edge(vertices.get(from),
                                          vertices.get(to),
                                          weight));
  }

  public boolean check_vertex(Vertex at)
  {
    if ( at == null ) return false;
    if ( vertices.get(at.get_index()) == null )
         return false;
    return true;
  }

  public Iterable<Vertex> get_vertices()
  {
    return vertices;
  }

  public String toString()
  {
    String s;
    int i;

    s = "";
    for (i = 0; i < vertices.size(); i++) {
      if (vertices.get(i) == null) {
        continue;
      }
      s += i + ": " + vertices.get(i).toString() + "\n";
    }

    return s;
  }
};
