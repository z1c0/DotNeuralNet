using System.Linq;
using System.Collections.Generic;

namespace DotNeuralNet
{
  public class Network
  {
    public Network(int inputNodeCount, int hiddenNodeCount, int outputNodeCount)
    {
      InputNodes = Enumerable.Range(0, inputNodeCount).Select(_ => new InputNode()).ToList();
      HiddenNodes = Enumerable.Range(0, hiddenNodeCount).Select(_ => new HiddenNode()).ToList();
      OutputNodes = Enumerable.Range(0, outputNodeCount).Select(_ => new OutputNode()).ToList();

      // Connect the nodes.
      foreach (var i in InputNodes)
      {
        foreach (var h in HiddenNodes)
        {
          h.AddIncoming(i);
        }
      }
      foreach (var h in HiddenNodes)
      {
        foreach (var o in OutputNodes)
        {
          o.AddIncoming(h);
        }
      }
    }

    public void Invalidate()
    {
      foreach (var n in OutputNodes)
      {
        n.Invalidate();
      }
      foreach (var n in HiddenNodes)
      {
        n.Invalidate();
      }
    }

    public IList<InputNode> InputNodes { get; private set;  }

    public IList<HiddenNode> HiddenNodes { get; private set; }

    public IList<OutputNode> OutputNodes { get; private set;  }
  }
}
