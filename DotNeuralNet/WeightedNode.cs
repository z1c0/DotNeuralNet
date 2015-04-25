using System.Linq;
using System.Collections.Generic;
using System;

namespace DotNeuralNet
{
  public abstract class WeightedNode : BaseNode
  {
    protected Func<double, double> _activationFunction;
    protected double? _value;

    public WeightedNode()
    {
      Incoming = new List<NodeLink>();
      Bias = Helpers.GetRandomWeight();
    }

    public double Bias { get; set; }

    internal void AddIncoming(BaseNode node)
    {
      Incoming.Add(new NodeLink(node) { Weight = Helpers.GetRandomWeight() });
    }

    internal void Invalidate()
    {
      _value = null;
    }

    public List<NodeLink> Incoming { get; private set; }

    public override double Value
    {
      get
      {
        if (!_value.HasValue)
        {
          var v = Bias + Incoming.Sum(i => i.Node.Value * i.Weight);
          _value = _activationFunction(v);
        }
        return _value.Value;
      }
      set
      {
        throw new InvalidOperationException();
      }
    }
  }
}
