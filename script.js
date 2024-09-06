function calculateScore() {
    let score = 0;

    // Get the selected values from each question
    score += parseInt(document.getElementById('q1').value);
    score += parseInt(document.getElementById('q2').value);
    score += parseInt(document.getElementById('q3').value);
    score += parseInt(document.getElementById('q4').value);
    score += parseInt(document.getElementById('q5').value);

    let resultText = "";

    // Simple scoring logic for feedback
    if (score >= 0 && score <= 4) {
        resultText = "Your mood seems normal, but always feel free to reach out if needed.";
    } else if (score >= 5 && score <= 9) {
        resultText = "You may be experiencing mild signs of depression.";
    } else if (score >= 10 && score <= 14) {
        resultText = "You may be experiencing moderate depression.";
    } else {
        resultText = "You may be experiencing severe depression. It might be a good idea to consult a healthcare provider.";
    }

    document.getElementById('result').innerText = resultText;
}

