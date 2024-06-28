#include "MusicSourceSep.h"

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

std::vector<torch::Tensor> musicSourceSeparation(const juce::AudioBuffer<float> buffer)
{
    std::vector<torch::Tensor> musicSourceSepRes;
    juce::AudioBuffer<float> audioBuffer = buffer;

    std::string modelFilePath = "../../../../../../../Resources/model_jit.pth";
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(modelFilePath);
        std::cout << "Model loaded successfully." << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }

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
    const int window_size = 485100;
    const int stride = 485100;

    std::vector<torch::Tensor> audioWindows = adjustAudioBufferToExpectedLength(audioBuffer, window_size, stride);
    printTensorShape(audioWindows[0], "audioWindows[0]");
    printTensorShape(audioWindows[1], "audioWindows[1]");

    int numTensors = audioWindows.size();
    std::vector<torch::Tensor> selectedParts;

    for (int i = 0; i < numTensors; ++i)
    {
        torch::Tensor audioTensor = audioWindows[i].unsqueeze(0); // Add batch dimension
        printTensorShape(audioTensor, "audioTensor");

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(audioTensor);
        torch::Tensor output;
        try
        {
            output = module.forward(inputs).toTensor(); // Model output
            std::cout << "Model inference completed successfully." << std::endl;
            printTensorShape(output, "Model output");

            selectedParts.push_back(output.select(1, 0).view({1, 2, 485100}));
        }
        catch (const c10::Error &e)
        {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
        }
    }

    torch::Tensor output = torch::cat(selectedParts, 2); // (1, 2, numTensors * 485100)
    std::cout << "Resulting tensor shape: " << output.sizes() << std::endl;

    output = output.index({"...", torch::indexing::Slice(0, numSamples)});
    printTensorShape(output, "Trimmed output");

    torch::Tensor drums = output[0];
    printTensorShape(drums, "drums tensor");

    if (drums.size(0) != 2) {
        std::cerr << "Error: The tensor does not have 2 channels." << std::endl;
    }

    for (int ch = 0; ch < drums.size(0); ++ch) {
        torch::Tensor channelTensor = drums.select(0, ch).unsqueeze(0);
        musicSourceSepRes.push_back(channelTensor);
    }
    printTensorShape(musicSourceSepRes[0], "musicSourceSepRes 0 tensor");
    printTensorShape(musicSourceSepRes[1], "musicSourceSepRes 1 tensor");



    return musicSourceSepRes;
}
