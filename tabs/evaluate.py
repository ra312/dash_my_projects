from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app

layout = [dcc.Markdown("""
### Final Results

Let us listen to what the model is telling !

|  Metric      |  Result   |
|--------------|-----------|
| Median Error |  0.49%    |
| Within 1%    | 14.69%    |
| Within 5 %   | 69.72%    |
| Within 10%   | 91.49%    |


"""),

html.Img(src='prediction_errors.png')]
