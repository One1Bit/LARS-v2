
#include <torch/torch.h>
#include <torch/script.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_core/juce_core.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Function to get an audio buffer from a file
juce::AudioBuffer<float> getAudioBufferFromFile(juce::File file, juce::AudioFormatManager &formatManager, double &sampleRate);

std::vector<torch::Tensor> adjustAudioBufferToExpectedLength(juce::AudioBuffer<float> &audioBuffer, int window_size, int stride);

void printTensorShape(const torch::Tensor &tensor, const std::string &name);

void printBufferShape(const juce::AudioBuffer<float> &buffer, const std::string &name);

std::vector<torch::Tensor> musicSourceSeparation(const juce::AudioBuffer<float> buffer);
