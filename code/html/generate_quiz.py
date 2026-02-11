import webbrowser
webbrowser.open("quiz.html")
questions = [
    {
        "question": "下列哪个是 Python 的关键字？",
        "options": ["function", "define", "def", "lambda"],
        "answer": 2,
        "explanation": "def 是定义函数的关键字。function 不是 Python 关键字。"
    }
]
html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>选择题练习</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        .question { margin-bottom: 30px; }
        .explanation { display: none; color: red; }
    </style>
</head>
<body>
    <h1>选择题练习</h1>
'''

for idx, q in enumerate(questions):
    html += f'''
    <div class="question" id="q{idx}">
        <p><b>{idx+1}. {q["question"]}</b></p>
    '''
    for i, opt in enumerate(q["options"]):
        html += f'''
        <button onclick="checkAnswer({idx}, {i})">{opt}</button><br>
        '''
    html += f'''
        <p class="explanation" id="exp{idx}">{q["explanation"]}</p>
    </div>
    '''

html += '''
<script>
    const answers = [''' + ','.join(str(q["answer"]) for q in questions) + '''];

    function checkAnswer(qid, selected) {
        const correct = answers[qid];
        const exp = document.getElementById("exp" + qid);
        if (selected == correct) {
            exp.style.color = 'green';
            exp.innerText = "✅ 正确！";
        } else {
            exp.style.color = 'red';
            exp.innerText = "❌ 错误！" + exp.innerText;
        }
        exp.style.display = "block";
    }
</script>
</body>
</html>
'''

with open("quiz.html", "w", encoding="utf-8") as f:
    f.write(html)
print("✅ 已生成 quiz.html")
