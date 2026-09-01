import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber = null;

self.addEventListener('message', async (event) => {
    const { audioData } = event.data;

    try {
        if (!transcriber) {
            self.postMessage({ status: 'loading', message: 'Загрузка модели Whisper в кэш...' });
            
            transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                quantized: true,
                progress_callback: (progress) => {
                    if (progress.status === 'downloading') {
                        const percent = Math.round((progress.loaded / (progress.total || 1)) * 100);
                        self.postMessage({ 
                            status: 'loading', 
                            message: `Скачивание модели: ${percent}%` 
                        });
                    }
                }
            });
        }

        self.postMessage({ status: 'processing', message: 'Идёт обработка нейросетью...' });

        // Явно передаём sampling_rate и упрощаем параметры, чтобы не вызывать зацикливание ONNX
        const output = await transcriber(audioData, {
            sampling_rate: 16000,
            language: 'russian',
            task: 'transcribe'
        });

        self.postMessage({ status: 'complete', text: output.text });

    } catch (error) {
        console.error(error);
        self.postMessage({ status: 'error', error: error.message || 'Ошибка обработки' });
    }
});
