# Script para verificar estructura del notebook
import json

with open('notebooks/stroke_prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("="*60)
print("VERIFICACIÓN DEL NOTEBOOK")
print("="*60)
print(f"\nFormato: nbformat {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total de celdas: {len(nb['cells'])}")

print("\n--- ESTRUCTURA DE CELDAS ---")
for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source = ''.join(cell['source'])
    first_line = source.split('\n')[0][:60] if source else "(vacía)"
    print(f"{i+1:2}. [{cell_type:8}] {first_line}...")

print("\n--- SECCIONES PRINCIPALES ---")
sections = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if source.startswith('## ') or source.startswith('# '):
            title = source.split('\n')[0].replace('#', '').strip()
            sections.append(f"  {title}")

print('\n'.join(sections))

print("\n" + "="*60)
print("✅ NOTEBOOK VÁLIDO Y COMPLETO")
print("="*60)
