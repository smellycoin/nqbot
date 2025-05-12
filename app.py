import os
import subprocess
import threading
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
import csv
from werkzeug.utils import secure_filename
import uuid
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'user_uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='backend_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    encoding='utf-8'
)
logging.debug('Flask app initialized and logging set up.')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    session_id = request.args.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        return redirect(url_for('index', session_id=session_id))

    # Store results and errors per session
    if 'results' not in app.config:
        app.config['results'] = {}
    if session_id not in app.config['results']:
        app.config['results'][session_id] = {'analysis_result': None, 'error_message': None, 'processing': False}

    analysis_result = app.config['results'][session_id]['analysis_result']
    error_message = app.config['results'][session_id]['error_message']
    processing = app.config['results'][session_id]['processing']
    form_data = request.form.to_dict()

    if request.method == 'POST':
        if 'chart_image' not in request.files:
            error_message = 'No chart image part'
            return render_template('index.html', error_message=error_message, form_data=form_data)
        
        file = request.files['chart_image']
        if file.filename == '':
            error_message = 'No selected chart image'
            return render_template('index.html', error_message=error_message, form_data=form_data)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create a unique filename to avoid overwrites and caching issues
            unique_filename = str(uuid.uuid4()) + "_" + filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(image_path)

            # Get parameters from the form
            timeframe = request.form.get('timeframe', '1h')
            ticker = request.form.get('ticker', 'NQ')
            use_llm = '--use_llm' if request.form.get('use_llm') else ''
            use_market_data = '--use_market_data' if request.form.get('use_market_data') else ''
            llm_model = request.form.get('llm_model', 'gpt2')
            extra_tips = '--extra_tips' if request.form.get('extra_tips') else ''
            user_context = request.form.get('user_context', '')

            # Construct the command to run nqbot.py
            # Ensure nqbot.py is in the same directory or provide the correct path
            nqbot_script_path = os.path.join(os.path.dirname(__file__), 'nqbot.py')
            if not os.path.exists(nqbot_script_path):
                error_message = f"Error: nqbot.py not found at {nqbot_script_path}. Ensure it's in the same directory as app.py."
                return render_template('index.html', error_message=error_message, form_data=form_data)

            command = [
                'python',
                nqbot_script_path,
                '--image', image_path,
                '--timeframe', timeframe,
                '--ticker', ticker,
                '--llm_model', llm_model,
                '--json_output' # Always use json_output for the web app
            ]
            if use_llm:
                command.append(use_llm)
            if use_market_data:
                command.append(use_market_data)
            if extra_tips:
                command.append(extra_tips)
            if user_context:
                command.extend(['--user_context', user_context])
            
            # Remove empty strings from command list (e.g. if flags are not set)
            command = [arg for arg in command if arg]

            app.config['results'][session_id]['processing'] = True
            app.config['results'][session_id]['error_message'] = None
            app.config['results'][session_id]['analysis_result'] = None

            # Run analysis in a separate thread
            thread = threading.Thread(target=run_analysis, args=(command, image_path, unique_filename, session_id))
            thread.start()

            return redirect(url_for('index', session_id=session_id))

    return render_template('index.html', analysis_result=analysis_result, error_message=error_message, form_data=form_data, session_id=session_id, processing=processing)


def run_analysis(command, image_path, unique_filename, session_id):
    try:
        logging.info(f"Starting analysis subprocess: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
        logging.debug(f"Subprocess finished. Return code: {process.returncode}")
        logging.debug(f"Subprocess STDOUT: {stdout}")
        logging.debug(f"Subprocess STDERR: {stderr}")
        current_analysis_result = None
        current_error_message = None
        if process.returncode != 0:
            current_error_message = f"Error running nqbot.py: {stderr}"
            logging.error(current_error_message)
            try:
                json_start = stdout.find('{')
                json_end = stdout.rfind('}') + 1
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    current_analysis_result = json.loads(stdout[json_start:json_end])
            except json.JSONDecodeError:
                current_error_message += "\nCould not parse JSON from nqbot.py output."
                logging.error("JSON decode error on error path.")
        else:
            json_start = stdout.find('{')
            json_end = stdout.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_start < json_end:
                try:
                    current_analysis_result = json.loads(stdout[json_start:json_end])
                    if current_analysis_result:
                        uploaded_image_template_path = os.path.join('user_uploads', unique_filename).replace('\\', '/')
                        current_analysis_result['image_path'] = uploaded_image_template_path
                except json.JSONDecodeError as e:
                    current_error_message = f"Failed to decode JSON from nqbot.py: {e}\nOutput was: {stdout}"
                    logging.error(current_error_message)
            else:
                current_error_message = "No JSON output found from nqbot.py. Ensure --json_output prints a valid JSON string to stdout."
                logging.error(current_error_message)
    except subprocess.TimeoutExpired:
        current_error_message = "nqbot.py script timed out after 5 minutes."
        logging.error(current_error_message)
    except Exception as e:
        current_error_message = f"An unexpected error occurred: {str(e)}"
        logging.exception(current_error_message)
    finally:
        app.config['results'][session_id]['analysis_result'] = current_analysis_result
        app.config['results'][session_id]['error_message'] = current_error_message
        app.config['results'][session_id]['processing'] = False
        logging.info(f"Analysis complete for session {session_id}. Processing flag set to False.")
        # Clean up uploaded image after processing if desired
        # if os.path.exists(image_path):
        #     os.remove(image_path)

@app.route('/status/<session_id>')
def get_status(session_id):
    if session_id in app.config.get('results', {}):
        return app.config['results'][session_id]
    return {"error": "Invalid session ID"}, 404


@app.route('/history')
def history():
    history_data = []
    history_file_path = os.path.join(os.path.dirname(__file__), 'trading_recommendations', 'trade_history.csv')
    logging.debug(f"Attempting to read history from: {history_file_path}")
    try:
        with open(history_file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_index, row in enumerate(reader):
                logging.debug(f"Processing row {row_index}: {row}")
                try:
                    row['confidence'] = float(row.get('confidence', 0.0)) if row.get('confidence') else 0.0
                except ValueError:
                    logging.warning(f"Could not convert confidence '{row.get('confidence')}' to float for row {row_index}. Defaulting to 0.0.")
                    row['confidence'] = 0.0
                
                raw_source_path = row.get('source', '')
                if raw_source_path.startswith('static/user_uploads/'):
                    row['image_path'] = raw_source_path[len('static/'):]
                elif raw_source_path.startswith('user_uploads/'):
                     row['image_path'] = raw_source_path
                elif raw_source_path.startswith('static/'): # Handles cases like 'static/some_other_folder/image.png'
                    row['image_path'] = raw_source_path[len('static/'):]
                elif raw_source_path: # If it's just a filename or relative path not starting with static/
                    base_name = os.path.basename(raw_source_path)
                    # Assumes images from various sources in CSV should end up in 'user_uploads' for display
                    row['image_path'] = os.path.join('user_uploads', base_name).replace('\\', '/')
                else:
                    row['image_path'] = ''

                row.setdefault('id', str(uuid.uuid4()))
                row.setdefault('timestamp', row.get('timestamp', 'N/A')) # Keep original if present
                row.setdefault('ticker', row.get('ticker', 'N/A'))
                row.setdefault('timeframe', row.get('timeframe', 'N/A'))
                row.setdefault('trend', row.get('trend', 'Neutral'))
                row.setdefault('recommendation', row.get('recommended_trade', 'Hold')) # Use 'recommended_trade' from CSV if exists
                row.setdefault('pattern_details', None) # CSV doesn't have this complex object
                
                history_data.append(row)
        logging.debug(f"Successfully loaded {len(history_data)} history items.")
        return jsonify(history_data)
    except FileNotFoundError:
        logging.error(f"History file not found: {history_file_path}")
        return jsonify({"error": "History data not found."}), 404
    except Exception as e:
        logging.exception(f"Error reading history file: {history_file_path}")
        return jsonify({"error": "Could not load history data."}), 500


if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    # Use threaded=True for handling multiple requests with threads
    app.run(debug=True, host='0.0.0.0', threaded=True)