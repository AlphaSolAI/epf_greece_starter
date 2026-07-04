import re, os

content_dir = 'content'
files = [
    'introduction.tex', 'conclusions.tex', 'dashboard.tex',
    'experiments.tex', 'feature_importance.tex',
    'literature.tex', 'methods.tex', 'data.tex',
]

def fix_text(text):
    # Fix \SI{VALUE}{\euro\per\MWh} -> VALUE~€/MWh
    text = re.sub(r'\\SI\{([0-9.,]+)\}\{\\euro\\per\\MWh\}', r'\1~€/MWh', text)
    # Fix Unicode right arrow → -> $\to$
    text = text.replace('\u2192', r'$\to$')
    # Fix Unicode star ⭐ -> [Best]
    text = text.replace('\u2b50', '[Best]')
    # Fix trophy 🏆
    text = text.replace('\U0001f3c6', '(1.)')
    # Fix → inside \texttt{} contexts (the above already handles →)
    return text

for fname in files:
    fpath = os.path.join(content_dir, fname)
    if not os.path.exists(fpath):
        print(f'SKIP: {fname}')
        continue
    with open(fpath, encoding='utf-8') as f:
        text = f.read()
    fixed = fix_text(text)
    if fixed != text:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f'FIXED: {fname}')
    else:
        print(f'OK: {fname}')
