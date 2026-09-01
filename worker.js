import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Отключаем попытки загрузки локальных моделей
env.allowLocalModels = false;

let transcriber = null;

self.addEventListener('message', async (event) => {
    const { audioData } = event.data;

    try {
        if (!transcriber) {
            self.postMessage({ status: 'loading', message: 'Загрузка модели Whisper...' });
            transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                quantized: true,
            });
        }

        self.postMessage({ status: 'processing', message: 'Распознавание текста...' });

        const output = await transcriber(audioData, {
            top_k: 0,
            do_sample: false,
            chunk_length_s: 30,
            stride_length_s: 5,
        });

        self.postMessage({ status: 'complete', text: output.text });
    } catch (error) {
        self.postMessage({ status: 'error', error: error.message });
    }
});