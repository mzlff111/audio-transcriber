import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0';

// Отключаем локальные проверки
env.allowLocalModels = false;

let transcriber = null;

self.addEventListener('message', async (event) => {
    const { audioData } = event.data;

    try {
        if (!transcriber) {
            self.postMessage({ status: 'loading', message: 'Загрузка легкой модели Whisper...' });
            
            // Загружаем оптимизированную модель
            transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                quantized: true,
            });
        }

        self.postMessage({ status: 'processing', message: 'Распознавание текста...' });

        // Разбиваем длинное аудио на небольшие порции, чтобы не перегружать память
        const output = await transcriber(audioData, {
            chunk_length_s: 20,
            stride_length_s: 3,
            language: 'russian',
            task: 'transcribe',
            return_timestamps: false
        });

        self.postMessage({ status: 'complete', text: output.text });
    } catch (error) {
        self.postMessage({ status: 'error', error: error.message });
    }
});
