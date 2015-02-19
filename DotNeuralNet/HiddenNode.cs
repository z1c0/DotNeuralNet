using System;
using System.Collections.Generic;

namespace DotNeuralNet
{
  public class HiddenNode : BaseNode
  {
    private readonly List<Tuple<BaseNode, double>> _incoming;

    public HiddenNode()
    {
      _incoming = new List<Tuple<BaseNode, double>>();
    }

    public void AddIncoming(BaseNode node)
    {
      _incoming.Add(new Tuple<BaseNode, double>(node, Helpers.GetRandomWeight()));
    }

    protected internal override double GetValue()
    {
      var value = .0;
      foreach (var n in _incoming)
      {
        value += n.Item1.GetValue() * n.Item2;
      }
      return value;
    }
  }
}
