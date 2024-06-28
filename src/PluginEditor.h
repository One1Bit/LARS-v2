/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

// not this:
//#include <JuceHeader.h>
// but this:
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cmath>
#include <JuceHeader.h>



#include "PluginProcessor.h"
#include "ClickableArea.h"



//==============================================================================
/**
*/



class DrumsDemixEditor  : public juce::AudioProcessorEditor,
                          // listen to buttons
                          public juce::Button::Listener,
                          // listen to AudioThumbnail
                          public juce::ChangeListener,
                          public juce::FileDragAndDropTarget,
                          private juce::Timer
                          

{
public:
    DrumsDemixEditor (DrumsDemixProcessor&);
    ~DrumsDemixEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    void buttonClicked(juce::Button* btn) override;

    juce::AudioBuffer<float> getAudioBufferFromFile(juce::File file);
    
    //juce::File Absolute = juce::File("/Users/alessandroorsatti/Documents/GitHub/DrumsDemix/drums_demix");
    juce::File absolutePath = juce::File::getCurrentWorkingDirectory().getParentDirectory();
    //juce::String Path = Absolute.getFullPathName();



    
    //VISUALIZER
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void thumbnailChanged();
    
    void displayOut(juce::AudioBuffer<float>& buffer, juce::AudioThumbnail& thumbnailOut);

    void paintIfNoFileLoaded(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, at::string Phrase);

    void paintIfFileLoaded(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorMusic(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorInput(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorKick(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorSnare(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorToms(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorHihat(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    void paintCursorCymbals(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

    void loadFile(const juce::String& path);

    //MODEL INFERENCE
    void InferModels(std::vector<torch::jit::IValue> my_input, torch::Tensor phase, int size);

    //CREATE WAV
    void CreateWavQuick(torch::Tensor yKickTensor, juce::String path, juce::String name); 
    void CreateWav(std::vector<at::Tensor> tList, juce::String name);



private:

    juce::String inputFileName;

    juce::File docsDir;
    juce::File filesDir;
    juce::File modelsDir;
    juce::File imagesDir;


    enum TransportState
    {
      Stopped,
      Starting,
      Stopping,
      Playing
    };

    TransportState state;
    
    juce::Image background;
    juce::Image logopoli;
    
    juce::ImageComponent imageMusic; //NEW
    juce::ImageComponent imageKit;
    juce::ImageComponent imageKick;
    juce::ImageComponent imageSnare;
    juce::ImageComponent imageToms;
    juce::ImageComponent imageHihat;
    juce::ImageComponent imageCymbals;
    juce::ImageComponent downloadIcon;
    juce::ImageComponent play;
    juce::ImageComponent stop;

    //buttons
    juce::ImageButton testButton;
    juce::ImageButton openButton;
    juce::ImageButton openMusicButton;
    bool musicSep = true;


    juce::ImageButton playButton;
    juce::ImageButton stopButton;

    juce::ImageButton downloadDrums;
    juce::ImageButton downloadKickButton;
    juce::ImageButton downloadSnareButton;
    juce::ImageButton downloadTomsButton;
    juce::ImageButton downloadHihatButton;
    juce::ImageButton downloadCymbalsButton;
    
    juce::ImageButton playMusicButton; //NEW
    juce::ImageButton stopMusicButton; //NEW

    juce::ImageButton playKickButton;
    juce::ImageButton stopKickButton;
    
    juce::ImageButton playSnareButton;
    juce::ImageButton stopSnareButton;
    
    juce::ImageButton playTomsButton;
    juce::ImageButton stopTomsButton;
    
    juce::ImageButton playHihatButton;
    juce::ImageButton stopHihatButton;
    
    juce::ImageButton playCymbalsButton;
    juce::ImageButton stopCymbalsButton;
    
    juce::ImageComponent musicImage; //NEW
    juce::ImageComponent kickImage;
    juce::ImageComponent snareImage;
    juce::ImageComponent tomsImage;
    juce::ImageComponent hihatImage;
    juce::ImageComponent cymbalsImage;
    juce::ImageComponent browseImage;
    juce::ImageComponent separate;

    ClickableArea areaDrums; //NEW
    ClickableArea areaKick;
    ClickableArea areaSnare;
    ClickableArea areaToms;
    ClickableArea areaHihat;
    ClickableArea areaCymbals;

    ClickableArea areaFull;

    //============================= NEW Interface start






    //============================ NEW Interface end

    
    //VISUALIZER
    juce::AudioThumbnail* thumbnailMusic; //NEW
    juce::AudioThumbnailCache* thumbnailCacheMusic; //NEW

    juce::AudioThumbnail* thumbnail;
    juce::AudioThumbnailCache* thumbnailCache;

    juce::AudioThumbnail* thumbnailKickOut;
    juce::AudioThumbnailCache* thumbnailCacheKickOut;
    
    juce::AudioThumbnail* thumbnailSnareOut;
    juce::AudioThumbnailCache* thumbnailCacheSnareOut;
    
    juce::AudioThumbnail* thumbnailTomsOut;
    juce::AudioThumbnailCache* thumbnailCacheTomsOut;
    
    juce::AudioThumbnail* thumbnailHihatOut;
    juce::AudioThumbnailCache* thumbnailCacheHihatOut;
    
    juce::AudioThumbnail* thumbnailCymbalsOut;
    juce::AudioThumbnailCache* thumbnailCacheCymbalsOut;

    
    //------------------------------------------------------------------------------------
    
    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> playSource;
    std::unique_ptr<juce::AudioFormatReaderSource> playMusic; //NEW
    std::unique_ptr<juce::MemoryAudioSource> playSourceKick;
    std::unique_ptr<juce::MemoryAudioSource> playSourceSnare;
    std::unique_ptr<juce::MemoryAudioSource> playSourceToms;
    std::unique_ptr<juce::MemoryAudioSource> playSourceHihat;
    std::unique_ptr<juce::MemoryAudioSource> playSourceCymbals;
    std::unique_ptr<juce::MemoryAudioSource> playSourceDrums; //NEW

    juce::File myFile;
    juce::File myFileOut;
    void transportStateChanged(TransportState newState, juce::String id);

    juce::AudioBuffer<float> bufferY;
    juce::AudioBuffer<float> bufferOut;

    std::vector<float> audioPoints;
    
    //audioPoints.call_back(new float (args));
    bool paintOut{ false };

    void timerCallback() override
    {
        repaint();
    }
    
    //load TorchScript modules:
    torch::jit::script::Module mymoduleKick;
    torch::jit::script::Module mymoduleSnare;
    torch::jit::script::Module mymoduleToms;
    torch::jit::script::Module mymoduleHihat;
    torch::jit::script::Module mymoduleCymbals;

    //output tensors
    at::Tensor yDrums; //NEW
    at::Tensor yKick;
    at::Tensor ySnare;
    at::Tensor yToms;
    at::Tensor yHihat;
    at::Tensor yCymbals;


    juce::Label textLabel;

    torch::Tensor fileTensor;

    
    

    
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    DrumsDemixProcessor& audioProcessor;

    //juce::ProgressBar progressBar{progress};

    //double currentPercentage{0};

    class ProgressThread : public juce::Thread
    {
    public:
        ProgressThread() : juce::Thread("Progress Thread")
        {

        }

        void run() override
        {
            //InferModels(my_input, stftFilePhase, fileTensor.sizes()[1]);
            progressincrement();
            signalThreadShouldExit();
        }

        //void InferModels(std::vector<torch::jit::IValue> my_input, torch::Tensor phase, int size)
        //{
        //    //c10::InferenceMode guard(true);
        //    DBG("Infering the Models...");
        //    Utils utils = Utils();
        //    //***INFER THE MODEL***


        //        //-Forward
        //    at::Tensor outputsKick = mymoduleKick.forward(my_input).toTensor();

        //    // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
        //    at::Tensor outputsSnare = mymoduleSnare.forward(my_input).toTensor();
        //    at::Tensor outputsToms = mymoduleToms.forward(my_input).toTensor();
        //    at::Tensor outputsHihat = mymoduleHihat.forward(my_input).toTensor();
        //    at::Tensor outputsCymbals = mymoduleCymbals.forward(my_input).toTensor();

        //    //-Need another dimension to do batch_istft
        //    outputsKick = torch::squeeze(outputsKick, 0);

        //    currentPercentage = currentPercentage + 0.2;

        //    DBG("outputs sizes: ");
        //    DBG(outputsKick.sizes()[0]);
        //    DBG(outputsKick.sizes()[1]);
        //    DBG(outputsKick.sizes()[2]);
        //    //DBG(outputs.sizes()[3]);




        //    // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
        //    outputsSnare = torch::squeeze(outputsSnare, 0);
        //    outputsToms = torch::squeeze(outputsToms, 0);
        //    outputsHihat = torch::squeeze(outputsHihat, 0);
        //    outputsCymbals = torch::squeeze(outputsCymbals, 0);


        //    //-Compute ISTFT

        //    yKick = utils.batch_istft(outputsKick, phase, size);

        //    DBG("y tensor sizes: ");
        //    DBG(yKick.sizes()[0]);
        //    DBG(yKick.sizes()[1]);


        //    // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING

        //    ySnare = utils.batch_istft(outputsSnare, phase, size);
        //    yToms = utils.batch_istft(outputsToms, phase, size);
        //    yHihat = utils.batch_istft(outputsHihat, phase, size);
        //    yCymbals = utils.batch_istft(outputsCymbals, phase, size);

        //    currentPercentage = currentPercentage + 0.2;



        //    /// RELOADARE I MODELLI E' UN MODO PER NON FAR CRASHARE AL SECONDO SEPARATE CONSECUTIVO, MA FORSE NON IL MIGLIOR MODO! (RALLENTA UN PO')


        //    try {
        //        //mymoduleKick=torch::jit::load("../src/scripted_modules/my_scripted_module_kick.pt");
        //        //juce::String kickString = modelsDir.getFullPathName() + "/my_scripted_module_kick.pt";
        //        //mymoduleKick = torch::jit::load(kickString.toStdString());
        //        std::stringstream modelStream;
        //        modelStream.write(BinaryData::my_scripted_module_kick_pt, BinaryData::my_scripted_module_kick_ptSize);
        //        mymoduleKick = torch::jit::load(modelStream);
        //        //MemoryInputStream* input = new MemoryInputStream(BinaryData::my_scripted_module_kick_pt, BinaryData::my_scripted_module_kick_ptSize, false);
        //    }
        //    catch (const c10::Error& e) {
        //        DBG("error"); //indicate error to calling code
        //    }


        //    try {
        //        //mymoduleSnare=torch::jit::load("../src/scripted_modules/my_scripted_module_snare.pt");
        //        //juce::String snareString = modelsDir.getFullPathName() + "/my_scripted_module_snare.pt";
        //        //mymoduleSnare = torch::jit::load(snareString.toStdString());
        //        std::stringstream modelStream;
        //        modelStream.write(BinaryData::my_scripted_module_snare_pt, BinaryData::my_scripted_module_snare_ptSize);
        //        mymoduleSnare = torch::jit::load(modelStream);
        //    }
        //    catch (const c10::Error& e) {
        //        DBG("error"); //indicate error to calling code
        //    }

        //    try {
        //        //mymoduleToms=torch::jit::load("../src/scripted_modules/my_scripted_module_toms.pt");
        //        //juce::String tomsString = modelsDir.getFullPathName() + "/my_scripted_module_toms.pt";
        //        //mymoduleToms = torch::jit::load(tomsString.toStdString());
        //        std::stringstream modelStream;
        //        modelStream.write(BinaryData::my_scripted_module_toms_pt, BinaryData::my_scripted_module_toms_ptSize);
        //        mymoduleToms = torch::jit::load(modelStream);
        //    }
        //    catch (const c10::Error& e) {
        //        DBG("error"); //indicate error to calling code
        //    }

        //    try {
        //        //mymoduleHihat=torch::jit::load("../src/scripted_modules/my_scripted_module_hihat.pt");
        //        //juce::String hihatString = modelsDir.getFullPathName() + "/my_scripted_module_hihat.pt";
        //        //mymoduleHihat = torch::jit::load(hihatString.toStdString());
        //        std::stringstream modelStream;
        //        modelStream.write(BinaryData::my_scripted_module_hihat_pt, BinaryData::my_scripted_module_hihat_ptSize);
        //        mymoduleHihat = torch::jit::load(modelStream);
        //    }
        //    catch (const c10::Error& e) {
        //        DBG("error"); //indicate error to calling code
        //    }

        //    try {
        //        //mymoduleCymbals=torch::jit::load("../src/scripted_modules/my_scripted_module_cymbals.pt");
        //        //juce::String cymbalsString = modelsDir.getFullPathName() + "/my_scripted_module_cymbals.pt";
        //        //mymoduleCymbals = torch::jit::load(cymbalsString.toStdString());
        //        std::stringstream modelStream;
        //        modelStream.write(BinaryData::my_scripted_module_cymbals_pt, BinaryData::my_scripted_module_cymbals_ptSize);
        //        mymoduleCymbals = torch::jit::load(modelStream);
        //    }
        //    catch (const c10::Error& e) {
        //        DBG("error"); //indicate error to calling code
        //    }

        //    currentPercentage = currentPercentage + 0.2;




        //}

        void progressincrement() {
            currentPercentage = currentPercentage + 0.2;
        }

        double currentPercentage{ 0 };
        std::unique_ptr<juce::ProgressBar> progress;
        //std::vector<torch::jit::IValue> my_input;


        //torch::jit::script::Module mymoduleKick;
        //torch::jit::script::Module mymoduleSnare;
        //torch::jit::script::Module mymoduleToms;
        //torch::jit::script::Module mymoduleHihat;
        //torch::jit::script::Module mymoduleCymbals;

        //at::Tensor yKick;
        //at::Tensor ySnare;
        //at::Tensor yToms;
        //at::Tensor yHihat;
        //at::Tensor yCymbals;

        //torch::Tensor stftFilePhase;

        //torch::Tensor fileTensor;

    };

    ProgressThread progressThread;
    

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumsDemixEditor)
};


