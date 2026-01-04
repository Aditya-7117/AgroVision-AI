const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const resultBox = document.getElementById("result");
const canvas = document.getElementById("confidenceChart");
const ctx = canvas.getContext("2d");

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (!file) {
        preview.style.display = "none";
        return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
    resultBox.innerHTML = "";
    clearChart();
});

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!imageInput.files.length) {
        resultBox.innerHTML = "Please upload an image first.";
        return;
    }

    resultBox.innerHTML = "Analyzing leaf image...";

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        resultBox.innerHTML = `
            <strong>Disease:</strong> ${data.disease}<br>
            <strong>Confidence:</strong> ${data.confidence}%<br><br>
            <strong>Recommended Actions:</strong>
            <ul>${data.remedies.map(r => `<li>${r}</li>`).join("")}</ul>
        `;

        drawChart(data.confidence);

    } catch (err) {
        resultBox.innerHTML = "Error analyzing image.";
        console.error(err);
    }
});

function drawChart(confidence) {
    clearChart();

    const center = canvas.width / 2;
    const radius = 70;
    const angle = (confidence / 100) * 2 * Math.PI;

    ctx.beginPath();
    ctx.arc(center, center, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 12;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(center, center, radius, -Math.PI / 2, angle - Math.PI / 2);
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 12;
    ctx.stroke();

    ctx.fillStyle = "#e5e7eb";
    ctx.font = "18px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(confidence + "%", center, center + 6);
}

function clearChart() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
