namespace DotNeuralNet.Perceptrons
{
  public class PerceptronTrainingRow
  {
    public PerceptronTrainingRow(double[] inputs, double expectedOutput)
    {
      Inputs = inputs;
      ExpectedOutput = expectedOutput;

    }
    public double[] Inputs { get; private set; }

    public double ExpectedOutput { get; private set; }
  }
}