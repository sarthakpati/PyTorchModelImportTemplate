#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "cbicaCmdParser.h"
#include "cbicaITKSafeImageIO.h"

#include "itkImage.h"
#include "itkCastImageFilter.h"


std::string inputImage, outputDirectory, trainedModelDirectory;

float sigma = 0.5;

typedef short                                 				PixelType;
const unsigned int Dimension = 3;
typedef itk::Image<PixelType, Dimension>      				ImageType;
typedef itk::ImageFileReader<ImageType>       				ReaderType;
typedef itk::ImageRegionIterator<ImageType> 			    IteratorType;
using DefaultImageType = itk::Image< float, ImageType::ImageDimension >;
using DefaultImageIteratorType = itk::ImageRegionIterator< DefaultImageType >;

//! Convert an ITK Image to a Tensor image
torch::Tensor itk2tensor(ImageType::Pointer itk_img) 
{
  // always cast to float
  auto caster = itk::CastImageFilter< ImageType, DefaultImageType >::New();
  caster->SetInput(itk_img);
  caster->Update();
  auto castedInput = caster->GetOutput();

  auto size = castedInput->GetLargestPossibleRegion().GetSize();
  std::cout << "Input Image Size: " << size << "\n";

  // convert array to tensor
  torch::Tensor tensor_img;
  tensor_img = torch::from_blob(castedInput->GetBufferPointer(),
    { 1, 1, (int)size[0], (int)size[1], (int)size[2] }, torch::kShort).clone();
  tensor_img = tensor_img.toType(torch::kFloat);
#if defined __CUDA_ARCH__
  //tensor_img = tensor_img.to(torch::kCUDA); // this doesn't seem to work, for whatever reason
#endif
  tensor_img = tensor_img.to(torch::kCPU);
  tensor_img.set_requires_grad(0);

  return tensor_img;
}

//! Convert a Tensor Image to an ITK image
ImageType::Pointer tensor2itk(torch::Tensor &t, ImageType::Pointer inputImage)
{
  std::cout << "tensor dtype = " << t.dtype() << std::endl;
  std::cout << "tensor size = " << t.sizes() << std::endl;
  t = t.toType(torch::kFloat);
  auto array = t.data< DefaultImageType::PixelType >();

  DefaultImageType::Pointer returnCastedImage;

  auto start = inputImage->GetLargestPossibleRegion().GetIndex();

  ImageType::SizeType  size;
  size[0] = t.size(2);
  size[1] = t.size(3);
  size[2] = t.size(4);

  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  returnCastedImage->SetRegions(region);
  returnCastedImage->Allocate();

  int len = size[0] * size[1] * size[2];

  DefaultImageIteratorType iter(returnCastedImage, returnCastedImage->GetRequestedRegion());
  int count = 0;
  // convert array to itk
  std::cout << "start!" << std::endl;
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter) 
  {
    auto temp = *array++;    //  ERROR!
    std::cout << temp << " ";
    iter.Set(temp);
    count++;
  }
  std::cout << "end!" << std::endl;

  // cast float image back to whatever the return type is supposed to be 
  auto caster = itk::CastImageFilter< DefaultImageType, ImageType >::New();
  caster->SetInput(returnCastedImage);
  caster->Update();

  return caster->GetOutput();
}


int main(int argc, const char* argv[]) {
  int a, b, c;
  if (argc != 4) {
    std::cerr << "usage: automyo input jitmodel output\n";
    return -1;
  }

  std::cout << "=========  jit start  =========\n";
  // 1. load jit script model
  std::cout << "Load script module: " << argv[2] << std::endl;
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[2]);
  module->to(at::kCUDA);

  // assert(module != nullptr);
  std::cout << "Load script module DONE" << std::endl;

  // 2. load input image
  const char* img_path = argv[1];
  std::cout << "Load image: " << img_path << std::endl;

  ReaderType::Pointer reader = ReaderType::New();

  if (!img_path) {
    std::cout << "Load input file error!" << std::endl;
    return false;
  }

  reader->SetFileName(img_path);
  reader->Update();

  std::cout << "Load image DONE!" << std::endl;

  ImageType::Pointer itk_img = reader->GetOutput();

  torch::Tensor tensor_img;
  if (!itk2tensor(itk_img, tensor_img)) {
    std::cerr << "itk2tensor ERROR!" << std::endl;
  }
  else {
    std::cout << "Convert array to tensor DONE!" << std::endl;
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor_img);

  // 3. predict by model
  torch::Tensor y = module->forward(inputs).toTensor();
  std::cout << "Inference DONE!" << std::endl;

  // 4. save the result to file
  torch::Tensor seg = y.gt(0.5);
  // std::cout << seg << std::endl;

  ImageType::Pointer out_itk_img = ImageType::New();
  if (!tensor2itk(seg, out_itk_img)) {
    std::cerr << "tensor2itk ERROR!" << std::endl;
  }
  else {
    std::cout << "Convert tensor to itk DONE!" << std::endl;
  }

  std::cout << out_itk_img << std::endl;

  return true;
}

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