// --- 常數設定 ---
const MODEL_PATH = 'resnet_model.onnx'; // 請確保這是 FP32 或 INT8 模型名稱
const INPUT_TENSOR_SIZE = 32 * 32 * 3;
const IMAGE_SIZE = 32;

const CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
];

// ⭐ 已更新為您 Python 腳本中的 CIFAR-10 標準化參數 ⭐
const NORM_MEAN = [0.4914, 0.4822, 0.4465];
const NORM_STD = [0.2470, 0.2435, 0.2616]; 


// --- DOM 元素快取 ---
const imageInput = document.getElementById('imageInput');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
const previewImg = document.getElementById('preview');

let inferenceSession = null;

/**
 * 步驟 1: 初始化 ONNX Runtime 會話並載入模型
 */
async function initializeModel() {
    statusDiv.textContent = '狀態: 正在載入模型...';
    try {
        // 設定 ONNX Runtime 的執行環境
        ort.env.wasm.numThreads = 1; 
        
        inferenceSession = await ort.InferenceSession.create(
            MODEL_PATH, 
            { executionProviders: ['wasm'] }
        );

        statusDiv.textContent = '狀態: 模型載入完成，可以上傳圖片。';
        imageInput.disabled = false;
    } catch (e) {
        console.error('模型載入失敗:', e);
        statusDiv.textContent = `狀態: 錯誤 - 模型載入失敗 (${e.message})，請檢查 ${MODEL_PATH} 是否存在於根目錄。`;
    }
}

/**
 * 步驟 2: 圖片前處理 (Resize, Normalization, HWC -> CHW)
 * @param {HTMLImageElement} imageElement 圖片元素
 * @returns {ort.Tensor} ONNX Runtime 格式的輸入張量
 */
function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(imageElement, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const data = imageData.data; 
    
    const floatData = new Float32Array(INPUT_TENSOR_SIZE); 
    let inputIndex = 0; 

    // 執行標準化和 HWC -> CHW 轉換 (與 Python np.transpose(2,0,1) 邏輯一致)
    for (let c = 0; c < 3; c++) { // 迴圈遍歷 R(0), G(1), B(2) 三個通道
        for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            
            // 獲取原始數據在 RGBA 陣列中的位置 (i*4 跳過像素，+c 選擇 R/G/B)
            const dataIndex = i * 4 + c; 
            
            // 1. [0, 255] 轉為 [0, 1]
            const normalized = data[dataIndex] / 255.0; 
            
            // 2. 應用標準化: (x - mean) / std
            const standardized = (normalized - NORM_MEAN[c]) / NORM_STD[c];
            
            floatData[inputIndex++] = standardized;
        }
    }

    // 創建 ONNX Runtime 張量 [1, C, H, W]
    // 假設 ONNX 模型的輸入名稱是 'input'
    const inputTensor = new ort.Tensor('float32', floatData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
    return inputTensor;
}


/**
 * 步驟 3: 處理圖片上傳並執行推理
 */
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file || !inferenceSession) return;

    statusDiv.textContent = '狀態: 圖片處理中...';
    resultDiv.innerHTML = '正在分析...'; 

    const reader = new FileReader();
    reader.onload = async (e) => {
        previewImg.src = e.target.result;
        
        const img = new Image();
        img.onload = async () => {
            try {
                // 1. 前處理
                const inputTensor = preprocessImage(img);
                
                statusDiv.textContent = '狀態: 正在執行 ONNX 推理...';
                
                // 2. 執行推理 
                // ⚠️ 這裡假設 ONNX 模型的輸入名稱是 'input'
                const feeds = { 'input': inputTensor }; 
                
                const results = await inferenceSession.run(feeds);
                
                // 3. 後處理
                const outputTensor = results[inferenceSession.outputNames[0]];
                const formattedResult = postprocessOutput(outputTensor.data);
                
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
 * 步驟 4: 後處理輸出張量 (Softmax 並格式化)
 * @param {Float32Array} outputData 模型的原始輸出數據 (logits)
 * @returns {string} 格式化的結果 HTML 字串
 */
function postprocessOutput(outputData) {
    let maxProbability = -Infinity;
    let predictedIndex = -1;
    
    // 計算 Softmax (使用 log-sum-exp 避免溢出)
    const logits = outputData;
    const probabilities = new Float32Array(logits.length);
    
    // 找到最大值
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxLogit) {
            maxLogit = logits[i];
        }
    }
    
    // 計算 Softmax
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
        probabilities[i] = Math.exp(logits[i] - maxLogit);
        sumExp += probabilities[i];
    }
    
    // 歸一化並找到最大機率
    for (let i = 0; i < logits.length; i++) {
        probabilities[i] /= sumExp;
        if (probabilities[i] > maxProbability) {
            maxProbability = probabilities[i];
            predictedIndex = i;
        }
    }

    // 格式化輸出
    const predictedClass = CIFAR10_CLASSES[predictedIndex];
    const confidence = (maxProbability * 100).toFixed(2);
    
    let html = `
        <h3>預測結果</h3>
        <p><strong>🥇 預測類別:</strong> <span style="color: green; font-weight: bold;">${predictedClass}</span></p>
        <p><strong>信心分數:</strong> ${confidence}%</p>
        <hr>
        <h4>Top 5 排名</h4>
    `;
    
    // 顯示前 5 名結果
    const sortedResults = Array.from(probabilities)
        .map((prob, index) => ({ prob, class: CIFAR10_CLASSES[index] }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5); 
        
    sortedResults.forEach(item => {
        html += `<p>${item.class}: ${(item.prob * 100).toFixed(2)}%</p>`;
    });

    return html;
}

// --- 啟動函式 ---
document.addEventListener('DOMContentLoaded', () => {
    if (imageInput) {
        imageInput.addEventListener('change', handleImageUpload);
        imageInput.disabled = true; 
    }
    
    // 啟動模型載入
    initializeModel();
});
