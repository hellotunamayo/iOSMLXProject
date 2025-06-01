//
//  ContentView.swift
//  MLXiOSProjApp
//
//  Created by Minyoung Yoo on 5/31/25.
//
import SwiftUI
import MLXLLM // Or MLXVLM for vision models
import MLXLMCommon
import MLX
import MarkdownUI

struct ContentView: View {
    @FocusState private var isFocused: Bool
    
    @State private var modelContainer: ModelContainer?
    @State private var isLoading = true
    @State private var isGenerating = false
    @State private var modelDownloadProgress: Double = 0.0
    @State private var modelDownloadProgressString: String = ""
    @State private var modelName: String = ""
    @State private var generatedText = ""
    @State private var prompt = ""
    
    let defaultModel: ModelConfiguration = ModelConfiguration(id: "mlx-community/Llama-3.2-1B-Instruct-bf16")
    
    var body: some View {
        NavigationStack {
            VStack {
                if isLoading {
                    Text(modelName)
                    ProgressView(value: modelDownloadProgress / 100)
                    Text(String(format: "%.1f%%", modelDownloadProgress))
                        .font(.title2)
                } else {
                    if isGenerating {
                        TextEditor(text: $generatedText)
                    } else {
                        ScrollView {
                            Markdown {
                                MarkdownContent(generatedText)
                            }
                            .markdownTheme(.docC)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    
                    HStack {
                        TextField("Ask something", text: $prompt)
                            .textFieldStyle(.roundedBorder)
                            .focused($isFocused)
                            .toolbar {
                                ToolbarItemGroup(placement: .keyboard) {
                                    Spacer()
                                    Button {
                                        isFocused = false
                                    } label: {
                                        Text("Close")
                                    }
                                }
                            }
                        
                        Button(action: {
                            if !isGenerating {
                                Task {
                                    withAnimation {
                                        self.isGenerating = true
                                    }
                                    await generateResponse(prompt: prompt)
                                    withAnimation {
                                        self.isGenerating = false
                                    }
                                }
                            }
                        }) {
                            Text(isGenerating ? "Generating..." : "Send")
                        }
                        .tint(isGenerating ? .gray : .accentColor)
                        .buttonStyle(.borderedProminent)
                    }
                }
            }
            .padding()
            .task {
                await loadModel()
            }
            .navigationTitle(modelName)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    func loadModel() async {
        isLoading = true
        do {
//            let config = ModelRegistry.llama3_2_1B_4bit
            let config = self.defaultModel
            self.modelContainer = try await LLMModelFactory.shared.loadContainer(configuration: config) { progress in
                Task { @MainActor in
                    let regex = try! NSRegularExpression(pattern: "mlx-community/")
                    let replacedModelName = regex.stringByReplacingMatches(in: config.name, range: NSRange(0..<config.name.count), withTemplate: "")
                    self.modelDownloadProgress = progress.fractionCompleted * 100
                    self.modelName = replacedModelName
                }
                #if DEBUG
                debugPrint("Downloading \(config.name): \(progress.fractionCompleted * 100)%")
                #endif
            }
            isLoading = false
        } catch {
            print("Error loading model: \(error)")
            isLoading = false
        }
    }
    
    func generateResponse(prompt: String) async {
        guard let container = modelContainer else { return }
        do {
            let input = UserInput(prompt: prompt)
            _ = try await container.perform { [input] context in
                let processedInput = try await context.processor.prepare(input: input)
                return try MLXLMCommon.generate(input: processedInput, parameters: .init(), context: context) { tokens in
                    // This closure is called as tokens are generated
                    let newText = context.tokenizer.decode(tokens: tokens)
                    Task { @MainActor in
                        self.generatedText = newText
                    }
                    return .more // Continue generating
                }
            }
            // You might process the final result here if needed
            self.prompt = ""
        } catch {
            print("Error during inference: \(error)")
        }
    }

}

#Preview {
    ContentView()
}
