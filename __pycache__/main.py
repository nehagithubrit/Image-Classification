
import detectionsugarcane as dft
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():

    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        image_path = 'static/files/'+file.filename # Path to your image
        label = dft.predict("static/files/"+file.filename)
        print(label)
        acc=round(dft.score[1],2)*100
        label1=str(acc)
        return render_template('result.htm',image_path=image_path,label=label,label1=acc)
        
    return render_template('home.html', form=form)
  
  

if __name__ == '__main__':
    app.run(debug=True)
