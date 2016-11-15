package edu.iastate.cs228.hw5;

/**
 * @author
 */

import java.util.HashMap;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class Dijkstra {

    /**
     * First, computes a shortest path from a source vertex to
     * a destination vertex in a graph by using Dijkstra's algorithm.
     * Second, visits and saves (in a stack) each vertex in the path,
     * in reverse order starting from the destination vertex,
     * by using the map object pred.
     * Third, uses a StringBuilder object to generate the return String
     * object by poping up the vertices from the stack;
     * the vertices in the String object are in the right order.
     * Note that the get_index() method is called from a Graph.Vertex object
     * to get its oringinal integer name.
     *
     * @param G
     *          - The graph in which a shortest path is to be computed
     * @param source
     *          - The first vertex of the shortest path
     * @param dest
     *          - The last vertex of the shortest path
     * @return a String object with three lines (separated by a newline character)
     *         such that line 1 shows the length of the shortest path,
     *         line 2 shows the cost of the path,
     *         and line 3 gives a list of the vertices (in the path)
     *         with a space between adjacent vertices.
     *
     *         The contents of an example String object:
     *         Path Length: 5
     *         Path Cost: 4
     *         Path: 0 4 2 5 7 9
     *
     * @throws NullPointerException
     *           - If any arugment is null
     *
     * @throws RuntimeException
     *           - If the given source or dest vertex is not in the graph
     *
     */
    public static String Dijkstra(Graph G, Graph.Vertex source, Graph.Vertex dest)
    {
      // TODO
    }

/**
 * A pair class with two components of types V and C, where
 * V is a vertex type and C is a cost type.
 */

private static class Vpair<V, C extends Comparable<? super C> > implements Comparable<Vpair<V, C>>
{
     private V  node;
     private C  cost;

     Vpair(V n, C c)
     {
       node = n;
       cost = c;
     }

     public V getVertex() { return node;}
     public C getCost() { return cost;}
     public int compareTo( Vpair<V, C> other )
     {
       return cost.compareTo(other.getCost() );
     }

     public String toString()
     {
       return "<" +  node.toString() + ", " + cost.toString() + ">";
     }

     public int hashCode()
     {
       return node.hashCode();
     }

     public boolean equals(Object obj)
     {
       if(this == obj) return true;
       if((obj == null) || (obj.getClass() != this.getClass()))
        return false;
       // object must be Vpair at this point
       Vpair<?, ?> test = (Vpair<?, ?>)obj;
       return
         (node == test.node || (node != null && node.equals(test.node)));
     }
}

}
