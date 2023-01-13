//===============================Q functions ==============
//JRR Web design experiment injecting JS code into the .html
//January 13 2023
//===============================Q functions ==============
master_list = [
'ID',
'today',
'dob',
'sex',
'ethnicity',
'indigenous',
'n_bedrooms',
'living_with_someone',
'n_outdoor_hours',
'n_indoor_per_week',
'avoided_crowded_places',
'avoided_contact_with_danger',
'had_covid_contact',
'had_covid_diagnosis',
'pos_test_#',
'hospitalized_due_to_covid',
'vaccinated_against_covid',
];


function add_vaccinated_against_covid()
{
	let vec = [
		"Yes, I've been vaccinated",
		"No, but I plan on being vaccinated",
		"No, I don't plan on being vaccinated",
		"No, I'm undecided about being vaccinated",
	];

	let ID = 'vaccinated_against_covid';
	let txt= 'Have you been vaccinated against COVID-19?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

}

function add_had_covid_diagnosis()
{
	let vec = [
		'Yes',
		'No',
		'Not sure',
	];

	let ID = 'had_covid_diagnosis';
	let txt= 'Have you been diagnosed with COVID-19?';
	let dd_obj = null;

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

	txt = 'What was the date of each positive ';
	txt += 'test result?[Leave blank if no positive test]';
	label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);

	let n_cases = 4;
	let ol_obj = $('<ol>');
	let li_obj = null;
	let div_obj = null;
	let cb_obj = null;
	let cb_id = null;
	let not_sure = ['Not sure'];

	for(let i = 1; i <= n_cases; i++)
	{
		ID  = 'pos_test_' + String(i);
		cb_id = ID + '_checkbox';
		txt = 'Test #' + String(i);
		li_obj = $('<li>');
		div_obj = create_div(ID);
		insert_calendar_to_div(ID, txt, div_obj);
		cb_obj = create_checkbox_from_list_and_ID(not_sure, cb_id);
		append_obj_to_div(cb_obj, '', div_obj);

		div_obj.appendTo(li_obj);
		li_obj.appendTo(ol_obj);
	}


	let hp_id = 'hospitalized_due_to_covid';
	txt = 'Were you hospitalized due to COVID-19?';

	li_obj = $('<li>');
	div_obj = create_div(hp_id);
	label_obj = create_label(hp_id, txt);
	append_obj_to_div(label_obj, '', div_obj);
	dd_obj = create_dropdown_from_list_and_ID(vec, hp_id);
	append_obj_to_div(dd_obj, '', div_obj);

	div_obj.appendTo(li_obj);
	li_obj.appendTo(ol_obj);

	ID = 'had_covid_diagnosis';
	append_obj_to_div(ol_obj, ID);

}

function add_had_covid_contact()
{
	let vec = [
		'Yes',
		'No',
		'Not sure',
	];


	let ID = 'had_covid_contact';
	let txt= 'Has a roommate, staff member, or visitor that ';
	txt += 'you"ve had close contact with had COVID-19 ';
	txt += '(high-risk close contact only)?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

}


function add_avoided_doing()
{
		let vec = [
			'NEVER',
			'RARELY',
			'OCCASSIONALY',
			'OFTEN',
			'ALWAYS',
				];

	let ID = 'avoided_doing';
	let txt= 'How many times have you done the following since March 2020?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);


	let list_of_ids = [
		'avoided_crowded_places',
		'avoided_contact_with_danger',
	];

	let list_of_txts = [
		'Avoided crowded places/gatherings',
		'Avoided contact with high risk people outside this home',
	];

	let dd_obj = null;

	let ol_obj = $('<ol>');
	let li_obj = null;
	let div_obj = null;

	for(let i = 0; i < list_of_ids.length; i++)
	{
		ID  = list_of_ids[i];
		txt = list_of_txts[i];
		li_obj = $('<li>');
		div_obj = create_div(ID);
		label_obj = create_label(ID, txt);
		div_obj.append(label_obj);
		dd_obj = create_dropdown_from_list_and_ID(vec, ID);
		div_obj.append(dd_obj);

		div_obj.appendTo(li_obj);
		li_obj.appendTo(ol_obj);
	}

	ID = 'avoided_doing';
	append_obj_to_div(ol_obj, ID);
}


function add_n_indoor_per_week()
{
	let vec = [];
	for(let i = 0; i <= 12; i++)
		vec.push(i)

	let ID = 'n_indoor_per_week';
	let txt= 'On average, how many times a week are you ';
	txt += 'indoors in a public place (e.g. community ';
	txt += 'centre, restaurant, coffee shop, grocery store, office, etc.)?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

}

function add_n_outdoor_hours()
{
	let vec = [];
	for(let i = 0; i <= 12; i++)
		vec.push(i)

	let ID = 'n_outdoor_hours';
	let txt= 'On average, how many hours do you spend ';
	txt += 'outside the retirement home in a typical day?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

}

function add_living_with_someone()
{
	let vec = [
		'I am alone',
		'Spouse',
		'Sibling',
		'Friend',
		'Prefer not to answer',
		'Other',
	];

	let ID = 'living_with_someone';
	let txt= 'If you are living with someone, ';
	txt += 'what is their relationship to you?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	include_other_for_dropdown(dd_obj);
	append_obj_to_div(dd_obj, ID);
	append_other_field_to_div(ID);

}

function add_n_bedrooms()
{
	let vec = [];
	for(let i = 0; i <= 6; i++)
		vec.push(i);

	let ID = 'n_bedrooms';
	let txt= 'How many bedrooms are in your retirement suite/apartment?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

}

function add_indigenous()
{
	let vec = [
		'Yes',
		'No',
		'Prefer not to answer',
	];

	let ID = 'indigenous';
	let txt= 'Are you an Indigenous person originating from North America?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	append_obj_to_div(dd_obj, ID);

}



function add_ethnicity()
{
	let vec = [
		'Caucasian',
		'South Asian',
		'Chinese',
		'Black',
		'Filipino',
		'Latin American',
		'Prefer not to answer',
		'Southeast Asian',
		'Arab',
		'West Asian ',
		'Korean',
		'Japanese',
		'Prefer to self-describe',
	];

	let ID = 'ethnicity';
	let txt= 'How would you describe your ethnicity or race?';
	txt += '(Select all that apply)';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let cb_obj = create_checkbox_from_list_and_ID(vec, ID);
	include_other_for_checkbox(cb_obj);
	append_obj_to_div(cb_obj, ID);
	append_other_field_to_div(ID);

}


function add_sex()
{
	let vec = [
		'Female',
		'Male',
		'Prefer not to answer',
		'Other',
	];

	let ID = 'sex';
	let txt= 'What was your sex at birth?';

	let label_obj = create_label(ID, txt);
	append_obj_to_div(label_obj, ID);
	let dd_obj = create_dropdown_from_list_and_ID(vec, ID);
	include_other_for_dropdown(dd_obj);
	append_obj_to_div(dd_obj, ID);
	append_other_field_to_div(ID);

}

function add_dob()
{
	let ID = 'dob';
	let q_txt = 'When were you born?'
	insert_calendar_to_div(ID, q_txt);
}

function add_today()
{
	let ID = 'today';
	let q_txt = "Today's date";
	insert_calendar_to_div(ID, q_txt);
}

//===============================Reusable functions ==============
function insert_calendar_to_div(ID, q_txt, div=null)
{
	let label_obj = create_label(ID, q_txt);
	let input_obj = $('<input type="date">').attr({id:ID});
	input_obj.appendTo(label_obj);
	if(div === null)
		$('#'+ID+'_div').append(label_obj);
	else
		div.append(label_obj);
}

function create_checkbox_from_list_and_ID(list, ID, name='')
{
	let checkboxes  = [];
	let checkbox_name = name;

	if (name === '')
		checkbox_name = ID;

	$.each(list, function(i, item)
		{
			let checkbox_id = ID + '_' + String(i);
			let label_obj = $('<label>').text(item);
			let checkbox_obj = $('<input type="checkbox">').attr({
				id:checkbox_id,
				name: checkbox_name,
			});

			checkbox_obj.appendTo(label_obj);
			checkboxes.push(label_obj);
		});

	return checkboxes;

}

function include_other_for_checkbox(cb_obj)
{

	/*
	let div_id = '#' + ID + '_div';
	let name = 'input[name="' + ID + '"]';
	let n_checkboxes= $(div_id).find(name).length;
	let last_checkbox_index = n_checkboxes - 1;
	let i = String(last_checkbox_index);
	let last_checkbox = '#' + ID + '_' + i
	*/
	let label = cb_obj.at(-1);
	let checkbox = label.children(':first');
	checkbox.click(function() {reveal_checkbox_other(this);});

}

function create_dropdown_from_list_and_ID(list, ID)
{
	let input_obj = $('<select>').attr({
		id: ID,
		name: ID,
	});

	$.each(list, function(i, item)
		{
			input_obj.append($('<option>', {
						value: i,
						text: item
			}));
		});

	return input_obj;
}

function include_other_for_dropdown(input_obj)
{
	input_obj.on('change', function(){reveal_dropdown_other(this);});
}

function append_other_field_to_div(ID, div = null)
{
	let other_obj = $('<input type="text">').attr({
		id:'other_' + ID,
		//placeholder:'Enter ' + ID,
		placeholder:'Enter description',
	});

	other_obj.css('width', 300);
	other_obj.css('display','none');

	if (div === null)
		append_obj_to_div(other_obj, ID);
	else
		div.append(other_obj);
}



function reveal_checkbox_other(checkbox)
{
	console.log(checkbox);
	let id = checkbox.id;
	id = id.substring(0, id.indexOf('_'));
	let input_id = '#other_' + id;
	let obj = $(input_id);

	if(checkbox.checked)
		obj.css('display', 'block');
	else
		obj.css('display', 'none');
}

function reveal_dropdown_other(selector)
{
	console.log(selector);
	let input_id = '#other_' + selector.id;
	let obj = $(input_id)[0];
	let last = selector.length - 1;
	if(selector.value == last)
		obj.style.display='block';
	else
		obj.style.display='none';
}

function create_label(ID, txt)
{
	let label_obj = $('<label>').text(txt).attr({
		'for': ID,
		id: ID + '_label',
	});
	return label_obj;
}

function create_div(ID)
{
	let div_obj = $('<div>').attr({
		'class': 'form_control',
		id: ID + '_div',
	});
	return div_obj;
}

function get_div_from_id(ID)
{
	let div_obj = $('#'+ID+'_div');
	return div_obj;
}

function append_obj_to_div(obj, ID, div=null)
{
	let div_obj = div;

	if(div === null)
		div_obj = get_div_from_id(ID);

	if (Array.isArray(obj))
		for (let i = 0; i < obj.length; i++)
			div_obj.append(obj[i]);
	else
			div_obj.append(obj);
}

//===============================End of Reusable functions ==============

//===============================MAIN==================================
add_today();
add_dob();
add_sex();
add_ethnicity();
add_indigenous();
add_n_bedrooms();
add_living_with_someone();
add_n_outdoor_hours();
add_n_indoor_per_week();
add_avoided_doing();
add_had_covid_contact();
add_had_covid_diagnosis();
add_vaccinated_against_covid();
