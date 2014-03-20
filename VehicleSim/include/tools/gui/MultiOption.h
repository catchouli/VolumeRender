#ifndef VEHICLESIM_MULTIOPTION
#define VEHICLESIM_MULTIOPTION

#include <Gwen/Controls/ComboBox.h>

#include "../../misc/GetterSetter.h"
#include "../../misc/Updatable.h"

namespace vlr
{
	template <typename baseType, typename setType>
	class MultiOption
		: public Gwen::Event::Handler, public Updatable, public OptionBase<baseType>
	{
	public:
		typedef setType type;

		typedef GetterSetter<baseType, type> GetterType;

		typedef type (baseType::*getterPointerType)() const;
		typedef void (baseType::*setterPointerType)(type);

		MultiOption(Gwen::Controls::Base* parent,
			GetterType getter)
			: _parent(parent), _getter(getter), OptionBase(getter.getBase())
		{
			init();
		}

		MultiOption(Gwen::Controls::Base* parent,
			baseType* base, type* pval)
			: _parent(parent), _getter(base, pval), OptionBase(base)
		{
			init();
		}

		MultiOption(Gwen::Controls::Base* parent, baseType* base,
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
			if (!_comboBox->HasFocus())
			{
				_comboBox->SelectItemByName(std::to_string(_getter.getValue()));
			}
		}

		void init()
		{
			_comboBox = new Gwen::Controls::ComboBox(_parent);
			_comboBox->onSelection.Add(this, &MultiOption::optionChanged);
		}

		Gwen::Controls::ComboBox* getComboBox()
		{
			return _comboBox;
		}

		void addOption(std::string text, type value)
		{
			Gwen::Controls::MenuItem* mu =
				_comboBox->AddItem(Gwen::UnicodeString(text.begin(), text.end()));
			
			type t = *new type(value);
			mu->UserData.Set("type", new type(value));
			mu->SetName(std::to_string(value));
		}

	protected:
		void optionChanged(Gwen::Event::Info info)
		{
			if (!_enabled)
				return;

			 if (_base == nullptr)
				 return;

			 Gwen::UserDataStorage* uds = &_comboBox->GetSelectedItem()->UserData;

			 if (uds->Exists("type"))
			 {
				 type value = *uds->Get<type*>("type");

				 _getter.setValue(value);
			 }
		}

	private:
		GetterType _getter;

		Gwen::Controls::Base* _parent;
		Gwen::Controls::ComboBox* _comboBox;
	};
}

#endif /* VEHICLESIM_MULTIOPTION */
