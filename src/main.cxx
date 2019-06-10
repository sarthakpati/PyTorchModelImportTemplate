#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "cbicaCmdParser.h"

std::string inputImage, outputDirectory, trainedModelDirectory;

float sigma = 0.5;

int main(int argc, const char* argv[]) 
{
  auto parser = cbica::CmdParser(argc, argv);
  parser.addRequiredParameter("m", "modelDir", cbica::Parameter::DIRECTORY, "Directory containing all required files for model", "The input trained model to be used");
  parser.addRequiredParameter("i", "inputImage", cbica::Parameter::FILE, "NIfTI input", "The input image to be processed");
  parser.addRequiredParameter("o", "outputDir", cbica::Parameter::DIRECTORY, "Dir with write access", "The output directory");
  parser.addOptionalParameter("s", "sigma", cbica::Parameter::FLOAT, "0-1", "Some random parameter shown as example", "Defaults to " + std::to_string(sigma));

  parser.getParameterValue("m", trainedModelDirectory);
  parser.getParameterValue("i", inputImage);
  parser.getParameterValue("o", outputDirectory);

  // Deserialize the ScriptModule from a file using torch::jit::load().
  auto module = torch::jit::load(trainedModelDirectory);

  assert(module != nullptr);
  std::cout << "All okay with model...\n";


}