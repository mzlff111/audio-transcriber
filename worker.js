import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Отключаем локальные модели и разрешаем загрузку с CDN
env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber = null;

self.addEventListener('message', async (event) => {
    const { audioData } = event.data;

    try {
        if (!transcriber) {
            self.postMessage({ status: 'loading', message: 'Загрузка нейросети Whisper (40 МБ)...' });
            
            // Инициализация с отслеживанием прогресса скачивания
            transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                quantized: true,
                progress_callback: (progress) => {
                    if (progress.status === 'downloading') {
                        const percent = Math.round((progress.loaded / progress.total) * 100) || 0;
                        self.postMessage({ 
                            status: 'loading', 
                            message: `Загрузка модели: ${percent}% (${progress.file})` 
                        });
                    }
                }
            });
        }

        self.postMessage({ status: 'processing', message: 'Идёт распознавание (это может занять до 1-2 мин)...' });

        // Запуск распознавания с разбиением на небольшие чанки по 15 секунд
        const output = await transcriber(audioData, {
            chunk_length_s: 15,
            stride_length_s: 2,
            language: 'russian',
            task: 'transcribe',
            return_timestamps: false,
            // Передаем статус каждые несколько секунд
            callback_function: (beams) => {
                self.postMessage({ status: 'processing', message: 'Анализ аудиопотока...' });
            }
        });

        self.postMessage({ status: 'complete', text: output.text });
    } catch (error) {
        console.error('Worker error:', error);
        self.postMessage({ status: 'error', error: error.message || 'Ошибка обработки нейросетью' });
    }
});
