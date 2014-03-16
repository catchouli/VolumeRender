#ifndef VEHICLESIM_BOOLOPTION
#define VEHICLESIM_BOOLOPTION

#include <Gwen/Controls/CheckBox.h>

#include "../../misc/GetterSetter.h"
#include "../../misc/Updatable.h"

namespace vlr
{
	template <typename baseType>
	class BoolOption
		: public Gwen::Event::Handler, public Updatable, public OptionBase<baseType>
	{
	public:
		typedef bool type;

		typedef type (baseType::*getterPointerType)() const;
		typedef void (baseType::*setterPointerType)(type);

		BoolOption(Gwen::Controls::Base* parent,
			baseType* base, type* pval)
			: _parent(parent), _getter(base, pval), OptionBase(base)
		{
			init();
		}

		BoolOption(Gwen::Controls::Base* parent, baseType* base,
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
			if (!_checkBox->HasFocus())
				_checkBox->SetChecked(_getter.getValue());
		}

		void init()
		{
			_checkBox = new Gwen::Controls::CheckBox(_parent);
			_checkBox->onCheckChanged.Add(this, &BoolOption::checkChanged);
		}

		Gwen::Controls::CheckBox* getCheckBox()
		{
			return _checkBox;
		}

	protected:
		void checkChanged(Gwen::Event::Info info)
		{
			if (!_enabled)
				return;

			 if (_base == nullptr)
				 return;

			 _getter.setValue(_checkBox->IsChecked());
		}

	private:
		GetterSetter<baseType, type> _getter;

		Gwen::Controls::Base* _parent;
		Gwen::Controls::CheckBox* _checkBox;
	};
}

#endif /* VEHICLESIM_BOOLOPTION */
