namespace DotNeuralNet
{
  public class BackPropagationTrainingRow
  {
    public BackPropagationTrainingRow(double[] inputs, double[] outputs)
    {
      Inputs = inputs;
      Outputs = outputs;
    }

    public double[] Inputs { get; private set; }

    public double[] Outputs { get; private set; }
  }
}