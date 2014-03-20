#ifndef VEHICLESIM_VECTOROPTION
#define VEHICLESIM_VECTOROPTION

#include <Box2D/Box2D.h>
#include <Gwen/Controls/TextBox.h>

#include "../../misc/GetterSetter.h"
#include "../../misc/Updatable.h"

namespace vlr
{
	template <typename baseType>
	class VectorOption
		: public Gwen::Event::Handler, public Updatable, public OptionBase<baseType>
	{
	public:
		typedef b2Vec2 type;

		typedef GetterSetter<baseType, type> GetterType;
		
		typedef type (baseType::*getterPointerType)() const;
		typedef const type& (baseType::*getterPointerTypeConstRefType)() const;
		typedef void (baseType::*setterPointerType)(type);
		typedef void (baseType::*setterPointerConstRefType)(const type&);

		VectorOption(Gwen::Controls::Base* parent,
			GetterType getter)
			: _parent(parent), _getter(getter), OptionBase(getter.getBase())
		{
			init();
		}

		VectorOption(Gwen::Controls::Base* parent,
			baseType* base, typename GetterType::finalPointerType pval)
			: _parent(parent), _getter(base, pval), OptionBase(base)
		{
			init();
		}

		VectorOption(Gwen::Controls::Base* parent, baseType* base,
			getterPointerType getterPointer, setterPointerType setterPointer)
			: _parent(parent), _getter(base, getterPointer, setterPointer), OptionBase(base)
		{
			init();
		}

		VectorOption(Gwen::Controls::Base* parent, baseType* base,
			getterPointerType getterPointer, setterPointerConstRefType setterPointer)
			: _parent(parent), _getter(base, getterPointer, setterPointer), OptionBase(base)
		{
			init();
		}

		VectorOption(Gwen::Controls::Base* parent, baseType* base,
			getterPointerTypeConstRefType getterPointer, setterPointerConstRefType setterPointer)
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
			if (!((Gwen::Controls::TextBox*)_textBoxX)->HasFocus())
				((Gwen::Controls::TextBox*)_textBoxX)->SetText(std::to_string(_getter.getValue().x));
			if (!((Gwen::Controls::TextBox*)_textBoxY)->HasFocus())
				((Gwen::Controls::TextBox*)_textBoxY)->SetText(std::to_string(_getter.getValue().y));
		}

		void init()
		{
			_textBoxX = new Gwen::Controls::TextBox(_parent);
			_textBoxX->onTextChanged.Add(this, &VectorOption::textChanged, new int(0));
			_textBoxY = new Gwen::Controls::TextBox(_parent);
			_textBoxY->onTextChanged.Add(this, &VectorOption::textChanged, new int(1));
		}

		Gwen::Controls::TextBox* getTextBoxX()
		{
			return _textBoxX;
		}

		Gwen::Controls::TextBox* getTextBoxY()
		{
			return _textBoxY;
		}

	protected:
		void textChanged(Gwen::Event::Info info)
		{
			if (!_enabled)
				return;

			 if (_base == nullptr)
				 return;

			float val;
			char buf[1024];
			int vals = sscanf(((Gwen::Controls::TextBox*)info.Control)->GetText().c_str(),
				"%f%s", &val, buf);

			if (vals == 1)
			{
				// Float value is valid
				((Gwen::Controls::TextBox*)info.Control)->
					SetTextColor(Gwen::Color(0, 0, 0));

				b2Vec2 vec = _getter.getValue();

				// Set value
				switch (*(int*)info.Data)
				{
				case 0:
					vec.x = val;
					_getter.setValue(vec);
					break;
				case 1:
					vec.y = val;
					_getter.setValue(vec);
					break;
				default:
					break;
				}
			}
			else
			{
				((Gwen::Controls::TextBox*)info.Control)->
					SetTextColor(Gwen::Color(255, 0, 0));
			}
		}

	private:
		GetterType _getter;

		Gwen::Controls::Base* _parent;
		Gwen::Controls::TextBox* _textBoxX;
		Gwen::Controls::TextBox* _textBoxY;
	};
}

#endif /* VEHICLESIM_VECTOROPTION */
