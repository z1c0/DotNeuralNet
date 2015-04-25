namespace DotNeuralNet
{
  public class InputNode : BaseNode
  {
    public override string ToString()
    {
      return Value.ToString();
    }

    public override double Value { get; set; }
  }
}
