namespace DotNeuralNet
{
  public class InputNode : BaseNode
  {
    protected internal override double GetValue()
    {
      return Value;
    }

    public double Value { get; set; }
  }
}
