package edu.iastate.cs228.hw5;

import java.io.FileInputStream;
import java.util.Scanner;
import java.util.InputMismatchException;
import java.util.Iterator;
import java.io.FileNotFoundException;

public class MazeSolver
{
  public static void main(String args[])
  { 
    try {
    Graph g = build_graph(args[0]);
    System.out.println("A graph representing a maze");
    System.out.println(g);

    Graph.Vertex source = get_endVertex(g, true);
    Graph.Vertex last = get_endVertex(g, false);
    String path = Dijkstra.Dijkstra(g, source, last);
    System.out.println("A shortest path in the maze");
    if ( path != null )
      System.out.println( path );
    else
      System.out.println( "No path exists" );
    }
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Constructs and returns a Graph object from the file with the given file name.
   * Each line in the file contains, in this order, a vertex, a colon (no space before the colon),
   * a number of vertex-cost pairs with a space before each pair.
   * Below is an example line:
   * 0: 2,3 4,1 6,2 8,4
   * This line specifies four directed edges from vertex 0 to each of
   * the four vertices: 2 (with a cost of 3), 4 (1), 6 (2), 8 (4).
   * If there is no edge from the vertex, then the line contains only
   * the vertex and a colon.
   *
   * Uses the g.add_edge() method to add each edge to the Graph object.
   *
   * @param fileName
   *            - The name of a file containing a directed graph
   * @throws FileNotFoundException
   *            - If the given file does not exist
   */
  public static Graph build_graph(String fileName) throws FileNotFoundException
  {
      FileInputStream f = new FileInputStream(fileName);
      Scanner l = new Scanner(f);
      Graph g = new Graph();

      while (l.hasNextLine()) {
        String str = l.nextLine();
        Scanner s = new Scanner(str);
        s.useDelimiter(":");
        int from = s.nextInt();
        s.skip("[: ]*");
        s.useDelimiter(",");
        while (s.hasNextInt()) {
          int to = s.nextInt();
          s.useDelimiter(" ");
          s.skip(",");
          int weight = s.nextInt();
          s.useDelimiter(",");
          if (s.hasNext()) {
            s.skip(" ");
          }
          g.add_edge(from, to, weight);
        }
      }
     return g;
  }

  /**
   *  Returns the first vertex in the given graph if isFirst is true,
   *  otherwise, returns the last vertex in the graph.
   *  Uses the get_vertices() method to get an Iterable<Graph.Vertex> object
   *  for getting access to each vertex in the given graph.
   *  Note that some Graph.Vertex references may be null, which
   *  needs to be skipped to avoid a NullPointerException. 
   *  This is due to the use of null to fill in the empty slots
   *  in the ArrayList<Graph.Vertex> object.
   *
   *  @param g
   *          - The Graph object
   *  @param isFirst
   *          - The flag to indicate whether the first or last vertex is returned
   *  @throws NullPointerException
   *          - If g is null
  */
  public static Graph.Vertex get_endVertex(Graph g, boolean isFirst)
  {
      if ( g == null )
           throw new NullPointerException();
      Iterator<Graph.Vertex> v = g.get_vertices().iterator();
      Graph.Vertex last = null;
      while ( v.hasNext() )
      {
        Graph.Vertex cur = v.next();
        if ( cur != null )
        {
          last = cur;
          if ( isFirst )
             return last;
        }
      }
     return last;
  }
}
