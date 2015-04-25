using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DotNeuralNet
{
  public class NodeLink
  {
    internal NodeLink(BaseNode node)
    {
      Node = node;
    }

    public override string ToString()
    {
      return Node + " * " + Weight;
    }

    internal BaseNode Node { get; private set; }

    public double Weight { get; set; }

    public double Delta { get; set; }
  }
}
