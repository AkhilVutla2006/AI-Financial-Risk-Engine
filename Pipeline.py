from parse_ais_pdf import parse_pdf_to_datapoint
from model_inference import explain_dataframe
from llm_modueles.llm import generate_explanation


def pipeline(data_file_path):
    print(f"Pipeline processing: {data_file_path}")
    
    df = parse_pdf_to_datapoint(data_file_path)

    annual_income = float(df['annual_income'].iloc[0]) if 'annual_income' in df.columns else 0.0
    
    val = explain_dataframe(df)
    
    val['annual_income'] = annual_income
    return val

def llm(val):
    stream = generate_explanation(val)
    return stream