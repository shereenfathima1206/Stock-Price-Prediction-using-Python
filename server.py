from flask import Flask, flash, render_template, redirect, url_for, request, session, jsonify 
#import json
from module.databasepkl import Database


from module.Db_user import Db_user
from module.Db_phonebook import Db_phonebook
#from module.Db_company import Db_company
#from module.Db_customer import Db_customer
#from module.Db_supplier import Db_supplier
#from module.Db_tax import Db_tax
#from module.Db_unit import Db_unit
#from module.Db_item import Db_item
 
app = Flask(__name__)
app.secret_key = "mys3cr3tk3y"


title="Stock Prediction"

# for phone book
dbUsr= Db_user();
dbPB = Db_phonebook();
#dbCMP = Db_company();
#dbCUS = Db_customer();
#dbSUP = Db_supplier();
#dbTAX = Db_tax();
#dbUNIT = Db_unit();
#dbITEM = Db_item();
db = Database()


# Index
@app.route('/')
def index():
    return render_template('index.html', title=title)


# User
@app.route('/showSignin')
def showSignin():
    if session.get('user'):
        data = dbPB.read(None)
        return render_template('signin.html', data = data,title=title)
       
    else:
        return render_template('signin.html', title=title)


@app.route('/showSignUp')
def showSignUp():
    return render_template('signup.html',title=title)

@app.route('/adduser', methods = ['POST', 'GET'])
def adduser():
    if request.method == 'POST' and request.form['save']:
        if dbUsr.insert(request.form):
            flash("A new user has been added")
        else:
            flash("A new user can not be added")
            
        return render_template('signin.html', title=title)
    else:
        return render_template('signin.html', title=title)

@app.route('/validateLogin',methods=['POST'])
def validateLogin():
    try:
        _username = request.form['inputEmail']
        _password = request.form['inputPassword']                                     
        data = dbUsr.readlogin(_username);
        if len(data) > 0:
            # if check_password_hash(str(data[0][3]),_password):
            if str(data[0][3])==_password:
                session['user'] = data[0][0]
                error =''
                #return redirect('/userHome')
                data = dbPB.read(None)    
                #return render_template('list_phonebook.html', data = data)
                return render_template('dsh.html', title=title)
            else:
                return render_template('error.html',error = 'Wrong Email address or Password.', title=title)
        else:
            return render_template('error.html',error = 'Wrong Email address or Password.', title=title)
            

    except Exception as e:
        return render_template('error.html',error = str(e), title=title)
    

@app.route('/userHome')
def userHome():
    if session.get('user'):
        return render_template('userHome.html')
    else:
        return render_template('error.html',error = 'Unauthorized Access', title=title)
    

@app.route('/logout')
def logout():
    session.pop('user',None)
    return redirect('/')


# Phone Book        

@app.route('/add/')
def add():
    return render_template('add_phonebook.html', title=title)

@app.route('/addphone', methods = ['POST', 'GET'])
def addphone():
    if request.method == 'POST' and request.form['save']:
        if dbPB.insert(request.form):
            flash("A new phone number has been added")
        else:
            flash("A new phone number can not be added")
            
        return redirect(url_for('showSignin'))
    else:
        return redirect(url_for('showSignin'))

@app.route('/update/<int:id>/')
def update(id):
    data = dbPB.read(id);
    
    if len(data) == 0:
        return redirect(url_for('showSignin'))
    else:
        session['update'] = id
        return render_template('update_phonebook.html', data = data, title=title)

@app.route('/updatephone', methods = ['POST'])
def updatephone():
    if request.method == 'POST' and request.form['update']:
        
        if dbPB.update(session['update'], request.form):
            flash('A phone number has been updated')
           
        else:
            flash('A phone number can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('showSignin'))
    else:
        return redirect(url_for('showSignin'))
    
@app.route('/delete/<int:id>/')
def delete(id):
    data = dbPB.read(id);
    
    if len(data) == 0:
        return redirect(url_for('showSignin'))
    else:
        session['delete'] = id
        return render_template('delete_phonebook.html', data = data)

@app.route('/deletephone', methods = ['POST'])
def deletephone():
    if request.method == 'POST' and request.form['delete']:
        
        if dbPB.delete(session['delete']):
            flash('A phone number has been deleted')
           
        else:
            flash('A phone number can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('showSignin'))
    else:
        return redirect(url_for('showSignin'))

# Dashboard
@app.route('/dashboard')
def dashboard():
    if session.get('user'):        
        data = dbPB.read(None)            
        return render_template('dsh.html', title=title)
    else:
        return render_template('signin.html', title=title)

 
#Company
@app.route('/listCompany')
def listCompany():
    if session.get('user'):        
        return render_template('dload.html', title=title)     
    else:
        return render_template('signin.html', title=title)

@app.route('/download', methods=['POST'])
def do_download():
    name = request.form['symbolname']
    data = db.download_quotes(name)
    return render_template('vw100.html', title=title)

@app.route('/listCustomer')
def listCustomer():
    if session.get('user'):        
        return render_template('headdata.html',title=title)        
    else:
        return render_template('signin.html', title=title)

@app.route('/head', methods=['POST'])
def do_head():
    name = request.form['symbolname']
    name1 = name +".csv"
    data = db.readhead(name1)
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na',name + ' Top 10 Share Values'], title=title)


@app.route('/listTop')
def do_top():
    data = db.PullData()
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na','News Articles Values'], title=title)

@app.route('/listTail')
def do_tails():
    data = db.AdjCloseData()
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na','News Articles Values'], title=title)

@app.route('/listDesc')
def do_Descs():
    data = db.NewColData()
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na','New Columns Added'], title=title)

@app.route('/listPred')
def rforestdata():
    if session.get('user'): 
        data = db.RandomForestPrediction()       
        return render_template('vviieeww.html',tables=[data.to_html(classes='male')],titles = ['na', ' Random Forest Predicted Values '], title=title)                
    else:
        return render_template('signin.html', title=title)

@app.route('/listRF')
def do_RF():
       
     return render_template('vw.html',titles = ['na', ' Random Forest Graph '], title=title)            

@app.route('/listLog')
def do_log():
    
    return render_template('vw1.html', titles = ['na','LR Graph'], title=title)


@app.route('/listMLP')
def do_MLP():
     
     return render_template('mlp.html',titles = ['na', ' MLP Classifier Graph '], title=title)       

@app.route('/listNB')
def do_NB():
           
     return render_template('nb.html',titles = ['na', ' Naive Bayes Graph '], title=title)   

         
@app.route('/listsvr')
def do_svr():
           
     return render_template('svr.html',titles = ['na', ' Naive Bayes Graph '], title=title)   
         




@app.route('/recommendation')
def do_recomend():
     data = db.recommend()       
     return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na','Stock Recommendation : Sell or Buy depending on the signal generated'], title=title)
 




#Supplier

@app.route('/lstm')
def lstm():
     data = db.MLPClassifierGraph()       
     return render_template('mlp.html',titles = ['na', ' MLP Classifier Graph '], title=title)


@app.route('/listSupplier')
def listtail():
    if session.get('user'):        
        return render_template('taildata.html', title=title)        
    else:
        return render_template('signin.html', title=title)

@app.route('/tail', methods=['POST'])
def do_tail():
    name = request.form['symbolname']
    name1 = name +".csv"
    data = db.readtail(name1)
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na',name + ' Last 10 Share Values'], title=title)

#Description
@app.route('/listTax')
def listdesc():
    if session.get('user'):        
        return render_template('descdata.html', title=title)        
    else:
        return render_template('signin.html', title=title)

@app.route('/desc', methods=['POST'])
def do_desc():
    name = request.form['symbolname']
    name1 = name +".csv"
    data = db.readdesc(name1)
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na',name + ' Last 10 Share Values'], title=title)





#Description
@app.route('/listMulti')
def listmulti():
    if session.get('user'):        
        return render_template('multi.html', title=title)        
    else:
        return render_template('signin.html', title=title)

@app.route('/listMulti', methods=['POST'])
def do_multi():

    
    data = db.readmulti()
    return render_template('view.html',tables=[data.to_html(classes='male')],titles = ['na', 'Comparison Share Values'], title=title)





@app.route('/addcmp/')
def addcmp():
    statedata=dbCMP.readstate("");
    return render_template('add_company.html', title=title, statedata = statedata)

@app.route('/addcompany', methods = ['POST', 'GET'])
def addcompany():
    if request.method == 'POST' and request.form['save']:
        if dbCMP.insert(request.form):
            flash("A new company has been added")
        else:
            flash("A new company can not be added")
            
        return redirect(url_for('listCompany'))
    else:
        return redirect(url_for('listCompany'))

@app.route('/updatecmp/<int:id>/')
def updatecmp(id):
    data = dbCMP.read(id);
    statedata=dbCMP.readstate("");    
    if len(data) == 0:
        return redirect(url_for('listCompany'))
    else:
        session['update'] = id
        return render_template('update_company.html', data = data, statedata = statedata, title=title)

@app.route('/updatecompany', methods = ['POST'])
def updatecompany():
    if request.method == 'POST' and request.form['update']:
        
        if dbCMP.update(session['update'], request.form):
            flash('A company has been updated')
           
        else:
            flash('A company can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('listCompany'))
    else:
        return redirect(url_for('listCompany'))
    
@app.route('/delcmp/<int:id>/')
def delcmp(id):
    data = dbCMP.read(id);
    statedata=dbCMP.readstate("");   
    if len(data) == 0:
        return redirect(url_for('listCompany'))
    else:
        session['delete'] = id
        return render_template('delete_company.html', data = data, statedata = statedata, title=title)

@app.route('/deletecompany', methods = ['POST'])
def deletecompany():
    if request.method == 'POST' and request.form['delete']:
        
        if dbCMP.delete(session['delete']):
            flash('A company has been deleted')
           
        else:
            flash('A company can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('listCompany'))
    else:
        return redirect(url_for('listCompany'))

#State Search Autocomplete
@app.route('/ac_state', methods=['GET','POST'])
def ac_state():
    search = request.args.get('term')        
    results = dbCMP.readstate(search)    
    return jsonify(results)

#Supplier Search Autocomplete
@app.route('/ac_supplier', methods=['GET','POST'])
def ac_supplier():
    search = request.args.get('term')        
    results = dbITEM.readsupplier(search)    
    return jsonify(results)

#Tax Search Autocomplete
@app.route('/ac_tax', methods=['GET','POST'])
def ac_tax():
    search = request.args.get('term')        
    results = dbITEM.readtax(search)    
    return jsonify(results)

#Unit Search Autocomplete
@app.route('/ac_unit', methods=['GET','POST'])
def ac_unit():
    search = request.args.get('term')        
    results = dbITEM.readunit(search)    
    return jsonify(results)


#Customer

    
@app.route('/addcus/')
def addcus():
    statedata=dbCUS.readstate("");
    return render_template('add_customer.html', title=title, statedata = statedata)

@app.route('/addcustomer', methods = ['POST', 'GET'])
def addcustomer():
    if request.method == 'POST' and request.form['save']:
        if dbCUS.insert(request.form):
            flash("A new customer has been added")
        else:
            flash("A new customer can not be added")
            
        return redirect(url_for('listCustomer'))
    else:
        return redirect(url_for('listCustomer'))

@app.route('/updatecus/<int:id>/')
def updatecus(id):
    data = dbCUS.read(id);
    statedata=dbCMP.readstate("");    
    if len(data) == 0:
        return redirect(url_for('listCustomer'))
    else:
        session['update'] = id
        return render_template('update_customer.html', data = data, statedata = statedata, title=title)

@app.route('/updatecustomer', methods = ['POST'])
def updatecustomer():
    if request.method == 'POST' and request.form['update']:
        
        if dbCUS.update(session['update'], request.form):
            flash('A customer has been updated')
           
        else:
            flash('A customer can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('listCustomer'))
    else:
        return redirect(url_for('listCustomer'))
    
@app.route('/delcus/<int:id>/')
def delcus(id):
    data = dbCUS.read(id);
    statedata=dbCUS.readstate("");   
    if len(data) == 0:
        return redirect(url_for('listCustomer'))
    else:
        session['delete'] = id
        return render_template('delete_customer.html', data = data, statedata = statedata, title=title)

@app.route('/deletecustomer', methods = ['POST'])
def deletecustomer():
    if request.method == 'POST' and request.form['delete']:
        
        if dbCUS.delete(session['delete']):
            flash('A customer has been deleted')
           
        else:
            flash('A customer can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('listCustomer'))
    else:
        return redirect(url_for('listCustomer'))


    
@app.route('/addsup/')
def addsup():
    statedata=dbSUP.readstate("");
    return render_template('add_supplier.html', title=title, statedata = statedata)

@app.route('/addsupplier', methods = ['POST', 'GET'])
def addsupplier():
    if request.method == 'POST' and request.form['save']:
        if dbSUP.insert(request.form):
            flash("A new supplier has been added")
        else:
            flash("A new supplier can not be added")
            
        return redirect(url_for('listSupplier'))
    else:
        return redirect(url_for('listSupplier'))

@app.route('/updatesup/<int:id>/')
def updatesup(id):
    data = dbSUP.read(id);
    statedata=dbCMP.readstate("");    
    if len(data) == 0:
        return redirect(url_for('listSupplier'))
    else:
        session['update'] = id
        return render_template('update_supplier.html', data = data, statedata = statedata, title=title)

@app.route('/updatesupplier', methods = ['POST'])
def updatesupplier():
    if request.method == 'POST' and request.form['update']:
        
        if dbSUP.update(session['update'], request.form):
            flash('A supplier has been updated')
           
        else:
            flash('A supplier can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('listSupplier'))
    else:
        return redirect(url_for('listSupplier'))
    
@app.route('/delsup/<int:id>/')
def delsup(id):
    data = dbSUP.read(id);
    statedata=dbSUP.readstate("");   
    if len(data) == 0:
        return redirect(url_for('listSupplier'))
    else:
        session['delete'] = id
        return render_template('delete_supplier.html', data = data, statedata = statedata, title=title)

@app.route('/deletesupplier', methods = ['POST'])
def deletesupplier():
    if request.method == 'POST' and request.form['delete']:
        
        if dbSUP.delete(session['delete']):
            flash('A supplier has been deleted')
           
        else:
            flash('A supplier can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('listSupplier'))
    else:
        return redirect(url_for('listSupplier'))     

#Tax
@app.route('/listTax')
def listTax():
    if session.get('user'):        
        data = dbTAX.read(None);    
        return render_template('list_tax.html', data = data,title=title)        
    else:
        return render_template('signin.html', title=title)
    
@app.route('/addtax/')
def addtax():    
    return render_template('add_tax.html', title=title)

@app.route('/addtaxs', methods = ['POST', 'GET'])
def addtaxs():
    if request.method == 'POST' and request.form['save']:
        if dbTAX.insert(request.form):
            flash("A new tax has been added")
        else:
            flash("A new tax can not be added")
            
        return redirect(url_for('listTax'))
    else:
        return redirect(url_for('listTax'))

@app.route('/updatetax/<int:id>/')
def updatetax(id):
    data = dbTAX.read(id);    
    if len(data) == 0:
        return redirect(url_for('listTax'))
    else:
        session['update'] = id
        return render_template('update_tax.html', data = data,  title=title)

@app.route('/updatetaxs', methods = ['POST'])
def updatetaxs():
    if request.method == 'POST' and request.form['update']:
        
        if dbTAX.update(session['update'], request.form):
            flash('A tax has been updated')
           
        else:
            flash('A tax can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('listTax'))
    else:
        return redirect(url_for('listTax'))
    
@app.route('/deltax/<int:id>/')
def deltax(id):
    data = dbTAX.read(id);
      
    if len(data) == 0:
        return redirect(url_for('listTax'))
    else:
        session['delete'] = id
        return render_template('delete_tax.html', data = data,  title=title)

@app.route('/deletetaxs', methods = ['POST'])
def deletetaxs():
    if request.method == 'POST' and request.form['delete']:
        
        if dbTAX.delete(session['delete']):
            flash('A tax has been deleted')
           
        else:
            flash('A tax can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('listTax'))
    else:
        return redirect(url_for('listTax'))


#Unit
@app.route('/listUnit')
def listUnit():
    if session.get('user'):        
        data = dbUNIT.read(None);    
        return render_template('list_unit.html', data = data,title=title)        
    else:
        return render_template('signin.html', title=title)
    
@app.route('/addunit/')
def addunit():    
    return render_template('add_unit.html', title=title)

@app.route('/addunits', methods = ['POST', 'GET'])
def addunits():
    if request.method == 'POST' and request.form['save']:
        if dbUNIT.insert(request.form):
            flash("A new unit has been added")
        else:
            flash("A new unit can not be added")
            
        return redirect(url_for('listUnit'))
    else:
        return redirect(url_for('listUnit'))

@app.route('/updateunit/<int:id>/')
def updateunit(id):
    data = dbUNIT.read(id);    
    if len(data) == 0:
        return redirect(url_for('listUnit'))
    else:
        session['update'] = id
        return render_template('update_unit.html', data = data,  title=title)

@app.route('/updateunits', methods = ['POST'])
def updateunits():
    if request.method == 'POST' and request.form['update']:
        
        if dbUNIT.update(session['update'], request.form):
            flash('A unit has been updated')
           
        else:
            flash('A unit can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('listUnit'))
    else:
        return redirect(url_for('listUnit'))
    
@app.route('/delunit/<int:id>/')
def delunit(id):
    data = dbUNIT.read(id);
      
    if len(data) == 0:
        return redirect(url_for('listUnit'))
    else:
        session['delete'] = id
        return render_template('delete_unit.html', data = data,  title=title)

@app.route('/deleteunits', methods = ['POST'])
def deleteunits():
    if request.method == 'POST' and request.form['delete']:
        
        if dbUNIT.delete(session['delete']):
            flash('A unit has been deleted')
           
        else:
            flash('A unit can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('listUnit'))
    else:
        return redirect(url_for('listUnit'))

#Item
@app.route('/listItem')
def listItem():
    if session.get('user'):        
        data = dbITEM.read(None);    
        return render_template('list_item.html', data = data,title=title)        
    else:
        return render_template('signin.html', title=title)
    
@app.route('/additem/')
def additem():
    supdata=dbITEM.readsupplier("");
    taxdata=dbITEM.readtax("");
    unidata=dbITEM.readunit("");    
    return render_template('add_item.html', title=title, supdata = supdata, taxdata = taxdata, unidata = unidata)

@app.route('/additems', methods = ['POST', 'GET'])
def additems():
    if request.method == 'POST' and request.form['save']:
        if dbITEM.insert(request.form):
            flash("A new item has been added")
        else:
            flash("A new item can not be added")
            
        return redirect(url_for('listItem'))
    else:
        return redirect(url_for('listItem'))

@app.route('/updateitem/<int:id>/')
def updateitem(id):
    data = dbITEM.read(id);
    supdata=dbITEM.readsupplier("");
    taxdata=dbITEM.readtax("");
    unidata=dbITEM.readunit(""); 
    if len(data) == 0:
        return redirect(url_for('listItem'))
    else:
        session['update'] = id
        return render_template('update_item.html', data = data,title=title, supdata = supdata, taxdata = taxdata, unidata = unidata)

@app.route('/updateitems', methods = ['POST'])
def updateitems():
    if request.method == 'POST' and request.form['update']:
        
        if dbITEM.update(session['update'], request.form):
            flash('A item has been updated')
           
        else:
            flash('A item can not be updated')
        
        session.pop('update', None)
        
        return redirect(url_for('listItem'))
    else:
        return redirect(url_for('listItem'))
    
@app.route('/delitem/<int:id>/')
def delitem(id):
    data = dbITEM.read(id);    
    if len(data) == 0:
        return redirect(url_for('listItem'))
    else:
        session['delete'] = id
        return render_template('delete_item.html', data = data , title=title)

@app.route('/deleteitem', methods = ['POST'])
def deleteitem():
    if request.method == 'POST' and request.form['delete']:
        
        if dbITEM.delete(session['delete']):
            flash('A item has been deleted')
           
        else:
            flash('A item can not be deleted')
        
        session.pop('delete', None)
        
        return redirect(url_for('listItem'))
    else:
        return redirect(url_for('listItem'))     

    
@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', title=title)

if __name__ == '__main__':
    app.run(debug = True, port=8181, host="0.0.0.0")
