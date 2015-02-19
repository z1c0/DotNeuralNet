using System;
using System.Collections.Generic;

namespace DotNeuralNet
{
  public class OutputNode : BaseNode
  {
    private readonly List<BaseNode> _incoming;
    private readonly Lazy<double> _value;

    public OutputNode()
    {
      _value = new Lazy<double>(GetValue);
      _incoming = new List<BaseNode>();
    }

    public void AddIncoming(BaseNode node)
    {
      _incoming.Add(node);
    }

    protected internal override double GetValue()
    {
      var value = .0;
      foreach (var n in _incoming)
      {
        value += n.GetValue();
      }
      return value;
    }

    public double Value => _value.Value;
  }
}
