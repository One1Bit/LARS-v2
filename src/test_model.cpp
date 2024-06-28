#include <torch/torch.h>
#include <torch/script.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_core/juce_core.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Function to get an audio buffer from a file
juce::AudioBuffer<float> getAudioBufferFromFile(juce::File file, juce::AudioFormatManager &formatManager, double &sampleRate)
{
    auto *reader = formatManager.createReaderFor(file);
    if (reader == nullptr)
    {
        throw std::runtime_error("Failed to create reader for audio file.");
    }
    sampleRate = reader->sampleRate;
    juce::AudioBuffer<float> audioBuffer;
    audioBuffer.setSize(reader->numChannels, static_cast<int>(reader->lengthInSamples));
    reader->read(&audioBuffer, 0, static_cast<int>(reader->lengthInSamples), 0, true, true);
    delete reader;
    return audioBuffer;
}

std::vector<torch::Tensor> adjustAudioBufferToExpectedLength(juce::AudioBuffer<float> &audioBuffer, int window_size, int stride)
{
    int numSamples = audioBuffer.getNumSamples();
    int numChannels = audioBuffer.getNumChannels();
    int num_windows = (numSamples + stride - 1) / stride; 

    std::vector<torch::Tensor> windows;
    for (int i = 0; i < num_windows; ++i)
    {
        int start = i * stride;
        int end = std::min(start + window_size, numSamples);

        std::vector<float> windowData;
        for (int ch = 0; ch < numChannels; ++ch) {
            const float* channelData = audioBuffer.getReadPointer(ch) + start;
            windowData.insert(windowData.end(), channelData, channelData + (end - start));

            // Check if this is the last window and needs padding for the current channel
            if ((end - start) < window_size) {
                int paddingSize = window_size - (end - start);
                windowData.insert(windowData.end(), paddingSize, 0.0f); // Zero padding for the current channel
            }
        }


        torch::Tensor tensor = torch::from_blob(windowData.data(), {numChannels, window_size}, torch::kFloat32).clone();
        std::cout << " shape tensor in adjustAudioBufferToExpectedLength: " << tensor.sizes() << std::endl;

        windows.push_back(tensor);
    }
    std::cout << " result of adjustAudioBufferToExpectedLength: " << windows.size() << std::endl;

    return windows;
}

void printTensorShape(const torch::Tensor &tensor, const std::string &name)
{
    std::cout << name << " shape: " << tensor.sizes() << std::endl;
}

void printBufferShape(const juce::AudioBuffer<float> &buffer, const std::string &name)
{
    std::cout << name << " shape: [" << buffer.getNumChannels() << ", " << buffer.getNumSamples() << "]" << std::endl;
}

int main()
{
    // Paths to input files and model
    std::string inputFilePath = "/Users/one/Downloads/__LARS/Test.wav";
    std::string modelFilePath = "/Users/one/Downloads/__LARS/Resources/model_jit.pth";
    std::string outputFilePath = "/Users/one/Downloads/__LARS/drums_output.wav";

    // Load the JIT model
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(modelFilePath);
        std::cout << "Model loaded successfully." << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // Initialize JUCE
    juce::ScopedJuceInitialiser_GUI libraryInitialiser;
    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    // Load the audio file into a JUCE audio buffer
    double sampleRate;
    juce::AudioBuffer<float> audioBuffer; 
    try
    {
        juce::File inputFile(inputFilePath);
        audioBuffer = getAudioBufferFromFile(inputFile, formatManager, sampleRate);
        std::cout << "Audio file loaded successfully." << std::endl;
        printBufferShape(audioBuffer, "Initial audioBuffer");
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Error loading audio file: " << std::endl;
        return -1;
    }

    // Ensure the audio buffer has 2 channels (stereo)
    if (audioBuffer.getNumChannels() != 2)
    {
        juce::AudioBuffer<float> stereoBuffer(2, audioBuffer.getNumSamples());
        stereoBuffer.copyFrom(0, 0, audioBuffer, 0, 0, audioBuffer.getNumSamples());
        if (audioBuffer.getNumChannels() == 1)
        {
            stereoBuffer.copyFrom(1, 0, audioBuffer, 0, 0, audioBuffer.getNumSamples());
        }
        audioBuffer = std::move(stereoBuffer);
        printBufferShape(audioBuffer, "Stereo audioBuffer");
    }

    int numSamples = audioBuffer.getNumSamples();
    const int expectedSamples = 485100;
    const int window_size = 485100;
    const int stride = 485100;

    // Adjust the audio buffer to the expected length
    std::vector<torch::Tensor> audioWindows = adjustAudioBufferToExpectedLength(audioBuffer, window_size, stride);
    printTensorShape(audioWindows[0], "audioWindows[0]");
    printTensorShape(audioWindows[1], "audioWindows[1]");
    //printTensorShape(audioWindows[2], "audioWindows[2]");


    torch::Tensor output;
    // Pass the audio through the model
    try
    {
        int numTensors=audioWindows.size();
        std::vector<torch::Tensor> selectedParts;
        for (int i=0;i<numTensors;i++){
            torch::Tensor audioTensor;
            audioTensor = torch::cat(audioWindows[i], 0).unsqueeze(0); // Add batch dimension
            printTensorShape(audioTensor, "audioTensor");

            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(audioTensor);
            output = module.forward(inputs).toTensor(); //model output

            std::cout << "Model inference completed successfully." << std::endl;
            printTensorShape(output, "Model output");
            
            // Select the first channel (e.g., drums) from the output tensor
            selectedParts.push_back(output.select(1, 0).view({1, 2, 485100}));

        }
        
        output = torch::cat(selectedParts, 2); // (1, 2, numTensors * 485100)
        std::cout << "Resulting tensor shape: " << output.sizes() << std::endl;

        //to the original size
        output = output.index({"...", torch::indexing::Slice(0, numSamples)});
        printTensorShape(output, "Trimmed output");
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error during model inference: " << e.what() << std::endl;
        return -1;
    }

    torch::Tensor drums = output[0]; 
    printTensorShape(drums, "drums tensor");

    // Create a JUCE audio buffer from the output tensor
    juce::AudioBuffer<float> drumsBuffer(2, drums.size(1)); // Ensure the buffer has 2 channels
    auto drumsAccessor = drums.accessor<float, 2>(); // Accessor
    for (int ch = 0; ch < drumsBuffer.getNumChannels(); ++ch)
    {
        float *channelData = drumsBuffer.getWritePointer(ch);
        for (int sample = 0; sample < drums.size(1); ++sample)
        {
            channelData[sample] = drumsAccessor[ch % drums.size(0)][sample];
        }
    }
    printBufferShape(drumsBuffer, "drumsBuffer");

    // Save the audio buffer to a file
    juce::File outputFile(outputFilePath);
    std::unique_ptr<juce::FileOutputStream> outputStream(outputFile.createOutputStream());
    if (outputStream != nullptr && outputStream->openedOk())
    {
        juce::WavAudioFormat wavFormat;
        std::unique_ptr<juce::AudioFormatWriter> writer(wavFormat.createWriterFor(outputStream.get(), sampleRate, drumsBuffer.getNumChannels(), 24, {}, 0));
        if (writer != nullptr)
        {
            if (!writer->writeFromAudioSampleBuffer(drumsBuffer, 0, drumsBuffer.getNumSamples()))
            {
                std::cerr << "Error writing audio sample buffer to file." << std::endl;
            }
            outputStream->flush(); // Ensure data is written to file
        }
        else
        {
            std::cerr << "Error creating audio format writer." << std::endl;
        }
    }
    else
    {
        std::cerr << "Error opening output stream to file: " << outputFile.getFullPathName() << std::endl;
    }

    std::cout << "Audio file saved successfully." << std::endl;
    return 0;
}
