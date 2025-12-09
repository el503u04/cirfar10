// --- 常數設定 ---
const MODEL_FP32_PATH = "resnet_exported.onnx";
const MODEL_INT8_PATH = "image_classifier_model_int8.onnx"; // 確保檔名一致!

const INPUT_TENSOR_SIZE = 32 * 32 * 3;
const IMAGE_SIZE = 32;

const CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
];

// 標準化參數 (與 Python 腳本一致)
const NORM_MEAN = [0.4914, 0.4822, 0.4465];
const NORM_STD = [0.2470, 0.2435, 0.2616]; 


// --- DOM 元素快取與 Sessions ---
const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
const previewImg = document.getElementById('preview');

let sessFP32 = null; // FP32 Session
let sessINT8 = null; // INT8 Session


/**
 * 步驟 1: 初始化 ONNX Runtime 會話並載入兩個模型
 */
async function initializeModel() {
    statusDiv.textContent = '狀態: 正在載入 FP32 與 INT8 模型...';
    try {
        ort.env.wasm.numThreads = 1; 

        // 載入 FP32 模型
        sessFP32 = await ort.InferenceSession.create(
            MODEL_FP32_PATH, 
            { executionProviders: ['wasm'] }
        );
        
        // 載入 INT8 模型
        sessINT8 = await ort.InferenceSession.create(
            MODEL_INT8_PATH, 
            { executionProviders: ['wasm'] }
        );

        statusDiv.textContent = '狀態: 兩模型載入完成，可以上傳圖片。';
        imageInput.disabled = false;
    } catch (e) {
        console.error('模型載入失敗:', e);
        statusDiv.textContent = `狀態: 嚴重錯誤 - 至少一個模型載入失敗 (${e.message})，請檢查檔名。`;
    }
}


// ----------------------------------------------------------------------
// 步驟 2: 圖片前處理 (與前一版相同，將圖片轉為張量)
// ----------------------------------------------------------------------
function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    
    const data = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data; 
    const floatData = new Float32Array(INPUT_TENSOR_SIZE); 
    let inputIndex = 0; 

    for (let c = 0; c < 3; c++) { 
        for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            const dataIndex = i * 4 + c; 
            const normalized = data[dataIndex] / 255.0; 
            const standardized = (normalized - NORM_MEAN[c]) / NORM_STD[c];
            floatData[inputIndex++] = standardized;
        }
    }
    // 假設 ONNX 輸入名稱為 'input'
    return new ort.Tensor('float32', floatData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

/**
 * 步驟 3 & 4: 推理與結果比較
 */
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file || !sessFP32 || !sessINT8) return;

    statusDiv.textContent = '狀態: 圖片處理中...';
    resultDiv.innerHTML = '正在分析...'; 

    const reader = new FileReader();
    reader.onload = async (e) => {
        previewImg.src = e.target.result;
        
        const img = new Image();
        img.onload = async () => {
            try {
                // 1. 前處理 (只需一次)
                const inputTensor = preprocessImage(img);
                const feeds = { 'input': inputTensor }; // ⚠️ 假設輸入名稱為 'input'
                
                statusDiv.textContent = '狀態: 正在執行雙模型推理...';
                
                // 2. FP32 推理
                const t0_fp32 = performance.now();
                const resFP32 = await sessFP32.run(feeds);
                const fp32_ms = (performance.now() - t0_fp32).toFixed(2);
                
                // 3. INT8 推理
                const t0_int8 = performance.now();
                const resINT8 = await sessINT8.run(feeds);
                const int8_ms = (performance.now() - t0_int8).toFixed(2);
                
                // 4. 後處理與比較
                const dataFP32 = resFP32[sessFP32.outputNames[0]].data;
                const dataINT8 = resINT8[sessINT8.outputNames[0]].data;
                
                const formattedResult = postprocessCompare(dataFP32, dataINT8, fp32_ms, int8_ms);
                
                statusDiv.textContent = '狀態: 推理完成。';
                resultDiv.innerHTML = formattedResult;
                
            } catch (error) {
                console.error('推理執行失敗:', error);
                resultDiv.innerHTML = `<strong>推理失敗!</strong> 錯誤訊息: ${error.message}`;
                statusDiv.textContent = `狀態: 錯誤 - 推理失敗。`;
            }
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

/**
 * 步驟 5: 後處理輸出張量並比較 (Softmax)
 * @param {Float32Array} outputDataFP32 FP32 Logits
 * @param {Float32Array} outputDataINT8 INT8 Logits
 * @param {string} fp32_ms FP32 推理時間 (ms)
 * @param {string} int8_ms INT8 推理時間 (ms)
 * @returns {string} 格式化的結果 HTML 字串
 */
function postprocessCompare(outputDataFP32, outputDataINT8, fp32_ms, int8_ms) {
    
    // --- 輔助函式: 計算 Softmax 並排序 ---
    function getTopK(logits, k = 3) {
        let maxLogit = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
            }
        }
        
        let sumExp = 0;
        const probabilities = new Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i] - maxLogit);
            sumExp += probabilities[i];
        }
        
        const results = Array.from(probabilities)
            .map((prob, index) => ({ prob: prob / sumExp, class: CIFAR10_CLASSES[index] }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, k);
            
        return results;
    }
    
    const top3FP32 = getTopK(outputDataFP32);
    const top3INT8 = getTopK(outputDataINT8);

    const speedup = (parseFloat(fp32_ms) / parseFloat(int8_ms)).toFixed(2);
    const topClassFP32 = top3FP32[0].class;
    const topClassINT8 = top3INT8[0].class;
    const classMatch = (topClassFP32 === topClassINT8) ? "✅ 相同" : "❌ 不同";


    let html = `
        <h3>📊 性能比較</h3>
        <p><strong>FP32 時間:</strong> ${fp32_ms} ms</p>
        <p><strong>INT8 時間:</strong> ${int8_ms} ms</p>
        <p><strong>加速比 (FP32/INT8):</strong> <span style="font-weight: bold; color: green;">${speedup}×</span></p>
        <p><strong>最高預測類別是否一致:</strong> ${classMatch}</p>
        <hr>
        
        <div style="display: flex; justify-content: space-between;">
            <div style="width: 48%;">
                <h4>FP32 (原始模型) Top 3</h4>
                ${top3FP32.map(item => 
                    `<p><strong>${item.class}:</strong> ${(item.prob * 100).toFixed(2)}%</p>`
                ).join('')}
            </div>
            <div style="width: 48%;">
                <h4>INT8 (量化模型) Top 3</h4>
                ${top3INT8.map(item => 
                    `<p><strong>${item.class}:</strong> ${(item.prob * 100).toFixed(2)}%</p>`
                ).join('')}
            </div>
        </div>
    `;

    return html;
}

// --- 啟動函式 ---
document.addEventListener('DOMContentLoaded', () => {
    // 確保圖片預覽區塊顯示正確
    const previewPlaceholder = document.getElementById('preview-placeholder');
    if (imageInput) {
        imageInput.addEventListener('change', handleImageUpload);
        imageInput.disabled = true; 
    }
    previewImg.onload = () => {
        previewImg.style.display = 'block';
        previewPlaceholder.style.display = 'none';
    };
    
    // 啟動模型載入
    initializeModel();
});
