import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber = null;

self.addEventListener('message', async (event) => {
    const { audioData } = event.data;

    try {
        if (!transcriber) {
            self.postMessage({ status: 'loading', message: 'Загрузка модели Whisper...' });
            
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

        // --- РУЧНОЕ РАЗБИЕНИЕ НА ЧАНКИ (по 10 секунд) ---
        const sampleRate = 16000;
        const chunkSizeSec = 10;
        const chunkSize = sampleRate * chunkSizeSec; // 160000 отсчетов
        const totalSamples = audioData.length;
        
        let fullText = '';
        let processedSamples = 0;

        while (processedSamples < totalSamples) {
            const end = Math.min(processedSamples + chunkSize, totalSamples);
            const chunk = audioData.slice(processedSamples, end);

            // Считаем и отправляем реальный процент готовности
            const percent = Math.round((end / totalSamples) * 100);
            self.postMessage({ 
                status: 'processing', 
                message: `Распознавание: ${percent}% (${Math.round(end / sampleRate)} сек. из ${Math.round(totalSamples / sampleRate)} сек.)` 
            });

            // Обрабатываем маленький чанк (не вызывает зависания)
            const output = await transcriber(chunk, {
                sampling_rate: sampleRate,
                language: 'russian',
                task: 'transcribe'
            });

            if (output && output.text) {
                fullText += output.text + ' ';
            }

            processedSamples += chunkSize;
            
            // Кроткий "отдых" для потока, чтобы браузер успел обновить интерфейс
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        self.postMessage({ status: 'complete', text: fullText.trim() });

    } catch (error) {
        console.error(error);
        self.postMessage({ status: 'error', error: error.message || 'Ошибка обработки' });
    }
});
