#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "cbicaCmdParser.h"
#include "cbicaITKSafeImageIO.h"

#include "itkImage.h"
#include "itkCastImageFilter.h"


std::string inputImageFile, outputDirectory, trainedModelDirectory;

float sigma = 0.5;

//! Convert an ITK Image to a Tensor image
template< class TImageType >
torch::Tensor itk2tensor(TImageType::Pointer itk_img) 
{
  using DefaultImageType = itk::Image< float, TImageType::ImageDimension >;
  using DefaultImageIteratorType = itk::ImageRegionIterator< DefaultImageType >;

  // always cast to float
  auto caster = itk::CastImageFilter< ImageType, DefaultImageType >::New();
  caster->SetInput(itk_img);
  caster->Update();
  auto castedInput = caster->GetOutput();

  auto size = castedInput->GetLargestPossibleRegion().GetSize();
  std::cout << "Input Image Size: " << size << "\n";

  // convert array to tensor
  torch::Tensor tensor_img;
  tensor_img = torch::from_blob(
    castedInput->GetBufferPointer(), // raw image pointer
    { 1, 1, (int)size[0], (int)size[1], (int)size[2] }, // image size - I wonder how this would work for N-D images?
    torch::kFloat).clone();
  tensor_img = tensor_img.toType(torch::kFloat); // default pixel type being passed around will be float
#if defined __CUDA_ARCH__
  //tensor_img = tensor_img.to(torch::kCUDA); // this doesn't seem to work, for whatever reason
#endif
  tensor_img = tensor_img.to(torch::kCPU);
  tensor_img.set_requires_grad(0);

  return tensor_img;
}

//! Convert a Tensor Image to an ITK image
template< class TImageType >
typename TImageType::Pointer tensor2itk(torch::Tensor &t, typename TImageType::Pointer inputImage)
{
  using DefaultImageType = itk::Image< float, TImageType::ImageDimension >;
  using DefaultImageIteratorType = itk::ImageRegionIterator< DefaultImageType >;

  std::cout << "tensor dtype = " << t.dtype() << std::endl;
  std::cout << "tensor size = " << t.sizes() << std::endl;
  t = t.toType(torch::kFloat);
  auto array = t.data< DefaultImageType::PixelType >();

  DefaultImageType::Pointer returnCastedImage;
  
  // get starting index of input image 
  auto start = inputImage->GetLargestPossibleRegion().GetIndex();

  // get size from tensor image 
  typename TImageType::SizeType  size;
  size[0] = t.size(2);
  size[1] = t.size(3);
  size[2] = t.size(4);

  auto size_itk = inputImage->GetLargestPossibleRegion().GetSize();
  // sanity check of tensor and itk image should happen here
  int tensorCounter = 2;
  for (size_t d = 0; d < ImageType::ImageDimension; d++)
  {
    if (size_itk[d] != t.size(tensorCounter))
    {
      std::cerr << "ITK Image size and Tensor image size mismatch.\n";
      return typename TImageType::New();
    }
    tensorCounter++;
  }

  typename TImageType::RegionType region;
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


int main(int argc, const char* argv[]) 
{
  auto parser = cbica::CmdParser(argc, argv);
  parser.addRequiredParameter("m", "modelDir", cbica::Parameter::DIRECTORY, "Directory containing all required files for model", "The input trained model to be used");
  parser.addRequiredParameter("i", "inputImage", cbica::Parameter::FILE, "NIfTI input", "The input image to be processed");
  parser.addRequiredParameter("o", "outputDir", cbica::Parameter::DIRECTORY, "Dir with write access", "The output directory");
  parser.addOptionalParameter("s", "sigma", cbica::Parameter::FLOAT, "0-1", "Some random parameter shown as example", "Defaults to " + std::to_string(sigma));

  parser.getParameterValue("m", trainedModelDirectory);
  parser.getParameterValue("i", inputImageFile);
  parser.getParameterValue("o", outputDirectory);

  // initialize the defaults for ITK to use, if needed
  using DefaultPixelType = float;
  using InputImageType = itk::Image< DefaultPixelType, 3 >;

  auto inputImage = cbica::ReadImage< InputImageType >(inputImageFile);

  // convert itk image to tensor
  auto convertedTensor = itk2tensor< InputImageType >(inputImage);

  // convert tensor image to itk
  auto convertedItkImage = tensor2itk< InputImageType >(convertedTensor, inputImage);

  // Deserialize the ScriptModule from a file using torch::jit::load().
  auto module = torch::jit::load(trainedModelDirectory);

  assert(module != nullptr);
  std::cout << "All okay with model...\n";


}