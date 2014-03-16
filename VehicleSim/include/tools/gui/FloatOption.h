#ifndef VEHICLESIM_FLOATOPTION
#define VEHICLESIM_FLOATOPTION

#include <Gwen/Controls/TextBox.h>

#include "OptionBase.h"
#include "../../misc/GetterSetter.h"
#include "../../misc/Updatable.h"

namespace vlr
{
	template <typename baseType>
	class FloatOption
		: public Gwen::Event::Handler, public Updatable, public OptionBase<baseType>
	{
	public:
		typedef float type;

		typedef type (baseType::*getterPointerType)() const;
		typedef void (baseType::*setterPointerType)(type);

		FloatOption(Gwen::Controls::Base* parent,
			baseType* base, type* pval)
			: _parent(parent), _getter(base, pval), OptionBase(base)
		{
			init();
		}

		FloatOption(Gwen::Controls::Base* parent, baseType* base,
			getterPointerType getterPointer, setterPointerType setterPointer)
			: _parent(parent), _getter(base, getterPointer, setterPointer), OptionBase(base)
		{
			init();
		}

		void setGetter(baseType* base, type* pval)
		{
			_getter = GetterSetter(base, pval);
		}

		void setGetter(baseType* type, getterPointerType getterPointer,
			setterPointerType setterPointer)
		{
			_getter = GetterSetter(base, getterPointer, setterPointer);
		}

		void update() override
		{
			if (!_enabled)
				return;

			_getter.setBase(_base);

			if (_base == nullptr)
				return;
			
			// Update textbox with value
			if (!((Gwen::Controls::TextBox*)_textBox)->HasFocus())
				((Gwen::Controls::TextBox*)_textBox)->SetText(std::to_string(_getter.getValue()));
		}

		void init()
		{
			_textBox = new Gwen::Controls::TextBox(_parent);
			_textBox->onTextChanged.Add(this, &FloatOption::textChanged);
			_textBox->onReturnPressed.Add(this, &FloatOption::textChanged);
		}

		Gwen::Controls::TextBox* getTextBox()
		{
			return _textBox;
		}

	protected:
		void textChanged(Gwen::Event::Info info)
		{
			if (!_enabled)
				return;

			 if (_base == nullptr)
				 return;

			type val;
			char buf[1024];
			int vals = sscanf(((Gwen::Controls::TextBox*)info.Control)->GetText().c_str(),
				"%f%s", &val, buf);

			if (vals == 1)
			{
				// Float value is valid
				_textBox->SetTextColor(Gwen::Color(0, 0, 0));

				// Set value
				_getter.setValue(val);
			}
			else
			{
				_textBox->SetTextColor(Gwen::Color(255, 0, 0));
			}
		}

	private:
		GetterSetter<baseType, type> _getter;

		Gwen::Controls::Base* _parent;
		Gwen::Controls::TextBox* _textBox;
	};
}

#endif /* VEHICLESIM_FLOATOPTION */
